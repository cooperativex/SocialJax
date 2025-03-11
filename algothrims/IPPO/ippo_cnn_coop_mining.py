""" 
Based on PureJaxRL & jaxmarl Implementation of PPO
"""
import os
import pickle
import time
from pathlib import Path
from typing import Sequence, NamedTuple

import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from PIL import Image
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import LogWrapper
from omegaconf import OmegaConf

import socialjax
from socialjax.wrappers.baselines import LogWrapper


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


def get_rollout(params, config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    done = False

    obs, state = env.reset(key_r)
    state_seq = [state]
    while not done:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *(env.observation_space()[0]).shape)

        pi, value = network.apply(params, obs_batch)
        action = pi.sample(seed=key_a0)
        env_act = unbatchify(
            action, env.agents, 1, env.num_agents
        )

        env_act = {k: v.squeeze() for k, v in env_act.items()}

        # STEP ENV
        obs, state, reward, done, info = env.step(key_s, state, env_act)
        done = done["__all__"]

        state_seq.append(state)

    return state_seq


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

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
            config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env, replace_info=False)

    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.,
        end_value=0.,
        transition_steps=config["REW_SHAPING_HORIZON"]
    )

    def linear_schedule(count):
        frac = (
                1.0
                - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
                / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):

        # INIT NETWORK
        network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))

        network_params = network.init(_rng, init_x)
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
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, update_step, rng = runner_state

                # jax.profiler.start_trace(".", True, True)

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4)).reshape(-1, *(env.observation_space()[0]).shape)
                # obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(-1, *env.observation_space().shape)

                print("input_obs_shape", obs_batch.shape)

                pi, value = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )

                # env_act = {k: v.flatten() for k, v in env_act.items()}
                env_act = [v for v in env_act.values()]

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                # shaped_reward = info.pop("shaped_reward")
                # current_timestep = update_step*config["NUM_STEPS"]*config["NUM_ENVS"]
                # reward = jax.tree_map(lambda x,y: x+y*rew_shaping_anneal(current_timestep), reward, shaped_reward)

                info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                transition = Transition(
                    batchify_dict(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                    info,
                )
                runner_state = (train_state, env_state, obsv, update_step, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, update_step, rng = runner_state
            last_obs_batch = jnp.stack([last_obs[:, a, ...] for a in env.agents]).reshape(-1, *(
                env.observation_space()[0]).shape)
            _, last_val = network.apply(train_state.params, last_obs_batch)

            def _calculate_gae(traj_batch, last_val):
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

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
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
                        train_state.params, traj_batch, advantages, targets
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
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            def callback(metric):
                wandb.log(metric)

            update_step = update_step + 1
            metric = jax.tree_map(lambda x: x.mean(), metric)
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
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
    n_agents = config["ENV_KWARGS"]["num_agents"]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    exp_name = f'{config["ENV_NAME"]}_{n_agents}agents_seed{config["SEED"]}_reward_{config["REWARD"]}_{timestamp}'

    wandb.init(
        config=config,
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=config["WANDB_TAGS"],
        mode=config["WANDB_MODE"],
        group=config["ENV_NAME"],
        job_type="train",
        name=exp_name,
        id=exp_name,
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = jax.jit(make_train(config))

    if config["PROFILING"]:
        # ---------------
        # START PROFILING
        # ---------------
        trace_dir = f"./jax_traces/train_trace_{exp_name}_{int(time.time())}"
        os.makedirs(trace_dir, exist_ok=True)

        # Start a JAX trace — only do this for a short portion of your code not to generate massive files.
        jax.profiler.start_trace(trace_dir, True, True)
        # ---------------

    out = jax.vmap(train_jit)(rngs)

    if config["PROFILING"]:
        # ---------------
        # STOP PROFILING
        # ---------------
        jax.profiler.stop_trace()
        # ---------------

    print("** Saving Results **")
    train_state = jax.tree_map(lambda x: x[0], out["runner_state"][0])
    save_path = f"./checkpoints/{exp_name}.pkl"
    save_params(train_state, save_path)
    params = train_state.params
    evaluate(params, socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]), config)

    if config["PROFILING"]:
        # Log the trace files to wandb
        for fn in os.listdir(trace_dir):
            if "events.out.trace" in fn:
                wandb.save(os.path.join(trace_dir, fn))


def save_params(train_state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    params = jax.tree_util.tree_map(lambda x: np.array(x), train_state.params)
    with open(save_path, 'wb') as f:
        pickle.dump(params, f)


def load_params(load_path):
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    return jax.tree_util.tree_map(lambda x: jnp.array(x), params)


def evaluate(params, env, config):
    rng = jax.random.PRNGKey(0)

    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng)
    returns = 0

    pics = []
    img = env.render(state)
    pics.append(img)
    root_dir = f"evaluation/{config['ENV_NAME']}/{config['REWARD']}"
    path = Path(root_dir + "/state_pics")
    path.mkdir(parents=True, exist_ok=True)
    network = ActorCritic(action_dim=env.action_space().n, activation=config["ACTIVATION"])

    for o_t in range(config["GIF_NUM_FRAMES"]):
        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *env.observation_space()[0].shape)
        pi, _ = network.apply(params, obs_batch)
        rng, _rng = jax.random.split(rng)
        actions = pi.sample(seed=_rng)

        env_act = {k: v.squeeze() for k, v in unbatchify(
            actions, env.agents, 1, env.num_agents
        ).items()}

        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(_rng, state, [v.item() for v in env_act.values()])

        returns += reward
        img = env.render(state)
        pics.append(img)

        print('###################')
        print(f'Actions: {env_act}')
        print(f'Reward: {reward}')
        print(f'Returns: {returns}')
        print(f'State: {state.agent_locs}')
        print("###################")

    print(f"Saving Episode GIF")
    pics = [Image.fromarray(np.array(img)) for img in pics]
    n_agents = len(env.agents)
    gif_path = f"{root_dir}/{n_agents}-agents_seed-{config['SEED']}_frames-{o_t + 1}.gif"
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
    wandb.log({"Episode GIF": wandb.Video(gif_path, caption="Evaluation Episode", format="gif")})


def tune(default_config):
    """
    Hyperparameter sweep with wandb, including logic to:
    - Initialize wandb
    - Train for each hyperparameter set
    - Save checkpoint
    - Evaluate and log GIF
    """
    import copy

    default_config = OmegaConf.to_container(default_config)

    sweep_config = {
        "name": "coop_mining",
        "method": "bayes",
        "metric": {
            "name": "returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {"values": [0.001, 0.0005, 0.0001, 0.00005]},
            "ACTIVATION": {"values": ["relu", "tanh"]},
            "UPDATE_EPOCHS": {"values": [2, 4, 8]},
            "NUM_MINIBATCHES": {"values": [4, 8, 16, 32]},
            "CLIP_EPS": {"values": [0.1, 0.2, 0.3]},
            "ENT_COEF": {"values": [0.001, 0.01, 0.1]},
            "NUM_STEPS": {"values": [64, 128, 256]},
        },
    }

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])
        # update the default params
        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v

        # Rename the run for clarity
        run_name = f"sweep_{config['ENV_NAME']}_seed{config['SEED']}"
        wandb.run.name = run_name
        print("Running experiment:", run_name)

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        outs = jax.block_until_ready(train_vjit(rngs))
        train_state = jax.tree_map(lambda x: x[0], outs["runner_state"][0])

        # Evaluate and log
        params = load_params(train_state.params)
        test_env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        evaluate(params, test_env, config)

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)


@hydra.main(version_base=None, config_path="config", config_name="ippo_cnn_coop_mining")
def main(config):
    if config["TUNE"]:
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()