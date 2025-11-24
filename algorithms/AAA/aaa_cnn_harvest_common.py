"""
AAA (Advantage Alignment Actor) Implementation
Based on: "Advantage Alignment Algorithms" (https://github.com/jduquevan/advantage-alignment)

Architecture: IPPO-style (independent networks per agent)
Key innovation: Modifies PPO advantage function to incorporate opponent advantages,
steering agents toward mutually beneficial equilibria.

AAA Formula: A*(s_t, a_t, b_t) = A¹(s_t, a_t, b_t) + β · aa_term(t)
Where:
- aa_term(t) = (Σ_{k<t} A¹(s_k, a_k, b_k)) · A²(s_t, a_t, b_t) / t
- A¹: Agent's own advantage at each timestep
- A²: Sum of opponent advantages (all other agents)
- β: Scaling parameter for opponent influence (AAA_BETA)
- t: Current timestep number (for normalization)

Note: The cumulative sum is a SIMPLE sum (not discounted by gamma).
The formula applies to timesteps t >= 1 (t=0 uses original advantage).
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import socialjax
from socialjax.wrappers.baselines import LogWrapper
import hydra
from omegaconf import OmegaConf
import wandb
import copy
import pickle
import os
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


class CNN(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        x = nn.Conv(
            features=32,
            kernel_size=(5, 5),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(
            features=64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = activation(x)

        return x


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = CNN(self.activation)(x)

        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[:, a] for a in agent_list])
    return x.reshape((num_actors, -1))


def batchify_dict(x: dict, agent_list, num_actors):
    x = jnp.stack([x[str(a)] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env, replace_info=False)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):

        # INIT NETWORK - IPPO style: one network per agent
        network = [
            ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
            for _ in range(env.num_agents)
        ]

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))

        network_params = [network[i].init(_rng, init_x) for i in range(env.num_agents)]

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        train_state = [
            TrainState.create(
                apply_fn=network[i].apply,
                params=network_params[i],
                tx=tx,
            )
            for i in range(env.num_agents)
        ]

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, update_step, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4))
                env_act = {}
                log_prob = []
                value = []
                for i in range(env.num_agents):
                    pi, value_i = network[i].apply(train_state[i].params, obs_batch[i])
                    action = pi.sample(seed=_rng)
                    log_prob.append(pi.log_prob(action))
                    env_act[env.agents[i]] = action
                    value.append(value_i)

                env_act = [v for v in env_act.values()]

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                transition = []
                done = [v for v in done.values()]
                for i in range(env.num_agents):
                    info_i = {
                        key: jax.tree_map(
                            lambda x: x.reshape((config["NUM_ACTORS"], 1)), value[:, i]
                        )
                        for key, value in info.items()
                    }
                    transition.append(
                        Transition(
                            done[i],
                            env_act[i],
                            value[i],
                            reward[:, i],
                            log_prob[i],
                            obs_batch[i],
                            info_i,
                        )
                    )
                runner_state = (train_state, env_state, obsv, update_step, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, update_step, rng = runner_state

            last_obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4))
            last_val = []
            for i in range(env.num_agents):
                _, last_val_i = network[i].apply(
                    train_state[i].params, last_obs_batch[i]
                )
                last_val.append(last_val_i)
            last_val = jnp.stack(last_val, axis=0)

            def _calculate_gae(traj_batch, last_val):
                """
                Calculate GAE advantages for all agents, then apply AAA modification.

                AAA modifies advantages using opponent advantages:
                A*(t) = A(t) + β * (cumsum(A[:t]) * sum_opponents(A(t))) / t
                """

                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                # Calculate standard GAE advantages for each agent
                advantages = []
                for i in range(env.num_agents):
                    _, advantages_i = jax.lax.scan(
                        _get_advantages,
                        (jnp.zeros_like(last_val[i]), last_val[i]),
                        traj_batch[i],
                        reverse=True,
                        unroll=16,
                    )
                    advantages.append(advantages_i)

                # Stack advantages: shape (num_agents, num_steps, num_envs)
                advantages = jnp.stack(advantages, axis=0)

                # IMPORTANT: Normalize advantages BEFORE applying AAA
                # This matches the original PyTorch implementation
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Apply AAA modification to each agent's advantages
                def apply_aaa_to_agent(agent_idx, advantages):
                    """
                    Apply AAA formula to a single agent's advantages.

                    Following original PyTorch implementation exactly:
                    A_1s = cumsum(A_s[:, :-1], dim=1)  # cumsum of [0:T-1]
                    A_2s = other agents' advantages at [1:T]
                    aa_terms = (A_1s * A_2s) / arange(1, T)
                    # Then prepend zero: [0, aa_terms]
                    """
                    # Get this agent's advantages: (num_steps, num_envs)
                    agent_adv = advantages[agent_idx]
                    num_steps = agent_adv.shape[0]

                    # A_1s: cumulative sum of advantages [0:T-1]
                    A_1s = jnp.cumsum(agent_adv[:-1, :], axis=0)  # (T-1, num_envs)

                    # A_2s: sum of opponent advantages at timesteps [1:T]
                    opponent_mask = jnp.ones(env.num_agents)
                    opponent_mask = opponent_mask.at[agent_idx].set(0)
                    A_2s = jnp.sum(
                        advantages[:, 1:, :] * opponent_mask[:, None, None], axis=0
                    )  # (T-1, num_envs)

                    # aa_terms = (A_1s * A_2s) / [1, 2, 3, ..., T-1]
                    time_steps = jnp.arange(1, num_steps, dtype=jnp.float32)[:, None]  # (T-1, 1)
                    aa_terms = (A_1s * A_2s) / time_steps  # (T-1, num_envs)

                    # Prepend zero at the beginning: timestep 0 has no AA modification
                    aa_terms_padded = jnp.concatenate(
                        [jnp.zeros((1, agent_adv.shape[1])), aa_terms], axis=0
                    )  # (T, num_envs)

                    # Apply AAA modification: A* = A + β * aa_terms
                    modified_adv = agent_adv + config["AAA_BETA"] * aa_terms_padded

                    return modified_adv

                # Apply AAA to all agents using vmap (parallel over agents)
                modified_advantages = jax.vmap(
                    apply_aaa_to_agent, in_axes=(0, None), out_axes=0
                )(jnp.arange(env.num_agents), advantages)

                # Convert back to list format for compatibility with IPPO structure
                modified_advantages_list = [
                    modified_advantages[i] for i in range(env.num_agents)
                ]
                targets_list = [
                    modified_advantages[i] + traj_batch[i].value
                    for i in range(env.num_agents)
                ]

                return modified_advantages_list, targets_list

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused, i):
                def _update_minbatch(train_state, batch_info, network_used):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets, network_used):
                        # RERUN NETWORK
                        pi, value = network_used.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets, network_used
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    lambda state, batch_info: _update_minbatch(
                        state, batch_info, network[i]
                    ),
                    train_state,
                    minibatches,
                )

                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state_dict = []
            metric = []
            for i in range(env.num_agents):
                update_state = (
                    train_state[i],
                    traj_batch[i],
                    advantages[i],
                    targets[i],
                    rng,
                )
                update_state, loss_info = jax.lax.scan(
                    lambda state, unused: _update_epoch(state, unused, i),
                    update_state,
                    None,
                    config["UPDATE_EPOCHS"],
                )
                update_state_dict.append(update_state)
                train_state[i] = update_state[0]
                metric_i = traj_batch[i].info
                metric_i["loss"] = loss_info[0]
                metric.append(metric_i)
                rng = update_state[-1]

            def callback(metric):
                wandb.log(metric)

            update_step = update_step + 1
            metric = jax.tree_map(lambda x: x.mean(), metric)
            for i in range(env.num_agents):
                metric[i]["update_step"] = update_step
                metric[i]["env_step"] = (
                    update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                )
            metric = metric[0]
            jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, update_step, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, 0, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


def single_run(config):
    config = OmegaConf.to_container(config)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["AAA", "ADVANTAGE_ALIGNMENT"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f'aaa_cnn_harvest_common',
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)

    print("** Saving Results **")
    filename = f'{config["ENV_NAME"]}_seed{config["SEED"]}_reward_{config["REWARD"]}'
    train_state = jax.tree_map(lambda x: x[0], out["runner_state"][0])

    params = []
    for i in range(config['ENV_KWARGS']['num_agents']):
        save_path = f"./checkpoints/aaa/{filename}_agent_{i}.pkl"
        save_params(train_state[i], save_path)
        params.append(load_params(save_path))

    print("** Evaluating Results **")
    evaluate(params, socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]), save_path, config)


def save_params(train_state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    params = jax.tree_util.tree_map(lambda x: np.array(x), train_state.params)

    with open(save_path, 'wb') as f:
        pickle.dump(params, f)


def load_params(load_path):
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    return jax.tree_util.tree_map(lambda x: jnp.array(x), params)


def evaluate(params, env, save_path, config):
    rng = jax.random.PRNGKey(0)

    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng)
    done = False

    pics = []
    img = env.render(state)
    pics.append(img)
    root_dir = f"evaluation/harvest_common"
    path = Path(root_dir + "/state_pics")
    path.mkdir(parents=True, exist_ok=True)

    for o_t in range(config["GIF_NUM_FRAMES"]):
        # Get observations for all agents
        obs_batch = jnp.stack([obs[a] for a in env.agents])
        env_act = {}
        network = [
            ActorCritic(action_dim=env.action_space().n, activation=config["ACTIVATION"])
            for _ in range(env.num_agents)
        ]
        for i in range(env.num_agents):
            obs_single = jnp.expand_dims(obs_batch[i], axis=0)
            pi, _ = network[i].apply(params[i], obs_single)
            rng, _rng = jax.random.split(rng)
            single_action = pi.sample(seed=_rng)
            env_act[env.agents[i]] = single_action

        # Execute actions
        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(
            _rng, state, [v.item() for v in env_act.values()]
        )
        done = done["__all__"]

        # Render
        img = env.render(state)
        pics.append(img)

        print('###################')
        print(f'Actions: {env_act}')
        print(f'Reward: {reward}')
        print("###################")

    # Save GIF
    print(f"Saving Episode GIF")
    pics = [Image.fromarray(np.array(img)) for img in pics]
    n_agents = len(env.agents)
    gif_path = f"{root_dir}/aaa_{n_agents}-agents_seed-{config['SEED']}_frames-{o_t + 1}.gif"
    pics[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        optimize=False,
        append_images=pics[1:],
        duration=200,
        loop=0,
    )

    # Log the GIF to WandB
    print("Logging GIF to WandB")
    wandb.log(
        {
            "Episode GIF": wandb.Video(
                gif_path, caption="AAA Evaluation Episode", format="gif"
            )
        }
    )


def tune(default_config):
    """
    Hyperparameter sweep with wandb
    """
    import copy

    default_config = OmegaConf.to_container(default_config)

    sweep_config = {
        "name": "aaa_harvest",
        "method": "grid",
        "metric": {
            "name": "returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            "AAA_BETA": {"values": [0.001, 0.005, 0.01, 0.05, 0.1]},
            "SEED": {"values": [30, 40, 50]},
        },
    }

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])
        config = copy.deepcopy(default_config)
        # only overwrite the single nested key we're sweeping
        for k, v in dict(wandb.config).items():
            if "." in k:
                parent, child = k.split(".", 1)
                config[parent][child] = v
            else:
                config[k] = v

        # Rename the run for clarity
        run_name = f"sweep_aaa_{config['ENV_NAME']}_beta{config['AAA_BETA']}_seed{config['SEED']}"
        wandb.run.name = run_name
        print("Running experiment:", run_name)

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        outs = jax.block_until_ready(train_vjit(rngs))
        train_state = jax.tree_map(lambda x: x[0], outs["runner_state"][0])

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)


@hydra.main(
    version_base=None, config_path="config", config_name="aaa_cnn_harvest_common"
)
def main(config):
    if config["TUNE"]:
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
