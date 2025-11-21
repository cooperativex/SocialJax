"""
AAA (Advantage Alignment Actor) Implementation
Based on: "Advantage Alignment Algorithms" (https://github.com/jduquevan/advantage-alignment)

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
from socialjax.wrappers.baselines import MAPPOWorldStateWrapper, LogWrapper
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
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        # x = nn.Conv(
        #     features=32,
        #     kernel_size=(3, 3),
        #     kernel_init=orthogonal(np.sqrt(2)),
        #     bias_init=constant(0.0),
        # )(x)
        # x = activation(x)
        # x = nn.Conv(
        #     features=32,
        #     kernel_size=(3, 3),
        #     kernel_init=orthogonal(np.sqrt(2)),
        #     bias_init=constant(0.0),
        # )(x)
        # x = activation(x)
        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(
            features=16, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = activation(x)

        return x


class Actor(nn.Module):
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        embedding = CNN(self.activation)(obs)
        actor_mean = nn.Dense(
            16, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)

        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        return pi


class Critic(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        world_state = x

        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = CNN(self.activation)(world_state)
        hidden = nn.Dense(
            features=16,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)

        hidden = activation(hidden)

        value = nn.Dense(
            features=1,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(hidden)
        # squeeze 去除最后一个维度
        return jnp.squeeze(value, axis=-1)
    

class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[str(a)] for a in agent_list])
    return x.reshape((num_actors, -1))

def batchify_numpy(x: dict, agent_list, num_actors):
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
    config["CLIP_EPS"] = config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"]

    env = MAPPOWorldStateWrapper(env)
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):

        # INIT NETWORK
        actor_network = Actor(env.action_space().n, activation=config["ACTIVATION"])
        critic_network = Critic(activation=config["ACTIVATION"])

        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        ac_init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))
            
        actor_network_params = actor_network.init(_rng_actor, ac_init_x)

        obs_shape = env.observation_space()[0].shape
        global_shape = (1, *obs_shape[:-1], obs_shape[-1] * env.num_agents)
        cr_init_x = jnp.zeros(global_shape)
        # cr_init_x = jnp.zeros((1, *(env.observation_space()[0]).shape)) 

        critic_network_params = critic_network.init(_rng_critic, cr_init_x)

        if config["ANNEAL_LR"]:            
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=actor_tx,
        )
        critic_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_done, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)


                obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *(env.observation_space()[0]).shape)
                # obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(-1, *env.observation_space().shape)

                # Removed print statement for JIT performance
                # print("input_obs_shape", obs_batch.shape)

                pi = actor_network.apply(train_states[0].params, obs_batch)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )

                # env_act = {k: v.flatten() for k, v in env_act.items()}
                env_act = [v for v in env_act.values()]

                #VALUE
                # World state is shared across all agents, only compute once per env
                world_state_per_env = jnp.transpose(last_obs, (0,2,3,1,4)).reshape(config["NUM_ENVS"], *(env.observation_space()[0]).shape[:-1], -1)
                # Compute value for each env (world state is same for all agents in same env)
                value_per_env = critic_network.apply(train_states[1].params, world_state_per_env)
                # Broadcast value to all agents in each env
                value = jnp.repeat(value_per_env, env.num_agents)


                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action,
                    value,
                    batchify_numpy(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                    world_state_per_env,  # Store compact world state (NUM_ENVS) instead of replicated (NUM_ACTORS)
                    info
                )
                runner_state = (train_states, env_state, obsv, done_batch, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, last_done, rng = runner_state

            # last_world_state = last_obs["world_state"].swapaxes(0,1)
            # last_world_state = last_world_state.reshape((config["NUM_ACTORS"],-1))
            last_world_state = jnp.transpose(last_obs, (0,2,3,1,4)).reshape(config["NUM_ENVS"], *(env.observation_space()[0]).shape[:-1], -1)
            last_val = critic_network.apply(train_states[1].params, last_world_state)
            last_val = jnp.expand_dims(last_val, axis=0)
            last_val = jnp.tile(last_val, (env.num_agents, 1))
            last_val = last_val.reshape((config["NUM_ACTORS"],-1))
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                # Standard GAE calculation first
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

                # AAA: Modify advantages using opponent advantages
                # Original implementation: aa_terms = (A_1s * A_2s) / timestep
                # where A_1s = cumsum(A_s[:, :-1]), A_2s = other_agents(A_s[:, 1:])
                # Shape: (num_steps, num_actors) where num_actors = num_agents * num_envs
                # Reshape to (num_steps, num_agents, num_envs) for easier agent separation
                num_steps = advantages.shape[0]
                advantages_reshaped = advantages.reshape(num_steps, env.num_agents, config["NUM_ENVS"])

                # Apply AAA formula for each agent
                def apply_aaa_to_agent(agent_idx, advantages_reshaped):
                    # Get this agent's advantages: (num_steps, num_envs)
                    agent_adv = advantages_reshaped[:, agent_idx, :]

                    # Following original implementation exactly:
                    # A_1s = torch.cumsum(A_s[:, :-1], dim=1) - cumsum of PAST advantages (exclude last timestep)
                    # A_s[:, 1:] - CURRENT advantages (exclude first timestep)

                    # Compute cumulative advantages for timesteps [0, T-2]
                    # These will align with current advantages at timesteps [1, T-1]
                    agent_adv_past = agent_adv[:-1, :]  # (T-1, num_envs)
                    cumulative_past_adv = jnp.cumsum(agent_adv_past, axis=0)  # (T-1, num_envs)

                    # Current advantages for timesteps [1, T-1]
                    agent_adv_current = agent_adv[1:, :]  # (T-1, num_envs)

                    # Get opponent advantages (sum of all other agents) for timesteps [1, T-1]
                    # A² = sum of all opponents' advantages
                    opponent_mask = jnp.ones(env.num_agents)
                    opponent_mask = opponent_mask.at[agent_idx].set(0)
                    opponent_adv = jnp.sum(
                        advantages_reshaped[1:, :, :] * opponent_mask[None, :, None],
                        axis=1
                    )  # (T-1, num_envs)

                    # AAA formula from original: (A_1s * A_2s) / timestep
                    # Timesteps are numbered 1, 2, 3, ..., T-1
                    time_steps = jnp.arange(1, num_steps, dtype=jnp.float32)[:, None]  # (T-1, 1)
                    aa_term = (cumulative_past_adv * opponent_adv) / time_steps

                    # Apply AAA modification with beta weight
                    # Original: A_s = A_s + aa_terms * aa_weight
                    modified_adv_current = agent_adv_current + config["AAA_BETA"] * aa_term

                    # For the first timestep (t=0), there's no cumulative past, so no AAA modification
                    # Prepend the original advantage for t=0
                    modified_adv = jnp.concatenate([
                        agent_adv[0:1, :],  # t=0, no modification
                        modified_adv_current  # t=1 to T-1, with AAA modification
                    ], axis=0)

                    return modified_adv

                # Apply to all agents using vmap (vectorized over agent_idx)
                # This is much faster than Python loop - compiles to parallel operations
                modified_advantages = jax.vmap(
                    apply_aaa_to_agent,
                    in_axes=(0, None),  # vmap over agent_idx, advantages_reshaped is broadcasted
                    out_axes=0  # output: (num_agents, num_steps, num_envs)
                )(jnp.arange(env.num_agents), advantages_reshaped)

                # Transpose to: (num_steps, num_agents, num_envs)
                modified_advantages = jnp.transpose(modified_advantages, (1, 0, 2))

                # Reshape back to original: (num_steps, num_actors)
                modified_advantages = modified_advantages.reshape(num_steps, config["NUM_ACTORS"])

                return modified_advantages, modified_advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    actor_train_state, critic_train_state = train_states
                    traj_batch, advantages, targets = batch_info

                    def _actor_loss_fn(actor_params, traj_batch, gae):
                        # RERUN NETWORK
                        pi = actor_network.apply(
                            actor_params,
                            jnp.reshape(traj_batch.obs, (-1, *(traj_batch.obs).shape[-3:])),
                        ) #.reshape(traj_batch.action.shape)

                        log_prob = pi.log_prob(traj_batch.action.reshape(-1))

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - jnp.reshape(traj_batch.log_prob, (-1,))
                        logratio = jnp.reshape(logratio, traj_batch.action.shape)
                        log_prob = jnp.reshape(log_prob, traj_batch.action.shape)
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
                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])
                        
                        actor_loss = (
                            loss_actor
                            - config["ENT_COEF"] * entropy
                        )
                        return actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)

                    def _critic_loss_fn(critic_params, traj_batch, targets):
                        # RERUN NETWORK
                        # world_state is now properly expanded and minibatched to match traj_batch.value
                        # Shape: (minibatch_size, H, W, C*num_agents)
                        world_state_batch = jnp.reshape(traj_batch.world_state, (-1, *(traj_batch.world_state).shape[-3:]))
                        value = critic_network.apply(critic_params, world_state_batch)
                        # Reshape to match traj_batch.value
                        value = jnp.reshape(value, traj_batch.value.shape)
                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, (value_loss)

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        actor_train_state.params, traj_batch, advantages
                    )
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params, traj_batch, targets
                    )
                    
                    actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
                    critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)
                    
                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[0],
                        "value_loss": critic_loss[0],
                        "entropy": actor_loss[1][1],
                        "ratio": actor_loss[1][2],
                        "approx_kl": actor_loss[1][3],
                        "clip_frac": actor_loss[1][4],
                    }
                    
                    return (actor_train_state, critic_train_state), loss_info

                (
                    train_states,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                # batch = (
                #     traj_batch,
                #     advantages.squeeze(),
                #     targets.squeeze(),
                # )
                # permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                # shuffled_batch = jax.tree_util.tree_map(
                #     lambda x: jnp.take(x, permutation, axis=1), batch
                # )
                
                # minibatches = jax.tree_util.tree_map(
                #     lambda x: jnp.swapaxes(
                #         jnp.reshape(
                #             x,
                #             [x.shape[0], config["NUM_MINIBATCHES"], -1]
                #             + list(x.shape[2:]),
                #         ),
                #         1,
                #         0,
                #     ),
                #     shuffled_batch,
                # )
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)

                # Expand world_state from per-env to per-actor for minibatching
                # This happens AFTER trajectory collection, so we still save memory during collection
                # Shape: (NUM_STEPS, NUM_ENVS, H, W, C) -> (NUM_STEPS, NUM_ACTORS, H, W, C)
                world_state_expanded = jnp.repeat(
                    jnp.expand_dims(traj_batch.world_state, axis=2),  # (NUM_STEPS, NUM_ENVS, 1, H, W, C)
                    env.num_agents,
                    axis=2
                ).reshape(config["NUM_STEPS"], config["NUM_ACTORS"], *traj_batch.world_state.shape[2:])

                # Replace world_state in traj_batch with expanded version
                traj_batch_expanded = Transition(
                    global_done=traj_batch.global_done,
                    done=traj_batch.done,
                    action=traj_batch.action,
                    value=traj_batch.value,
                    reward=traj_batch.reward,
                    log_prob=traj_batch.log_prob,
                    obs=traj_batch.obs,
                    world_state=world_state_expanded,
                    info=traj_batch.info
                )

                batch = (traj_batch_expanded, advantages, targets)
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

                train_states, _ = jax.lax.scan(
                    _update_minbatch, train_states, minibatches
                )
                update_state = (
                    train_states,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                # Return empty dict to avoid accumulating loss_info across epochs
                return update_state, {}

            update_state = (
                train_states,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, _ = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_states = update_state[0]
            metric = traj_batch.info
            # loss_info is not accumulated to save memory
            rng = update_state[-1]

            def callback(metric):
                wandb.log(metric)
            update_steps = update_steps + 1
            metric = jax.tree_map(lambda x: x.mean(), metric)
            metric["update_steps"] = update_steps
            metric["env_step"] = update_steps * config["NUM_STEPS"] * config["NUM_ENVS"]

            # jax.experimental.io_callback(callback, None, metric)

            jax.debug.callback(callback, metric)
            runner_state = (train_states, env_state, last_obs, last_done, rng)
            # Return empty dict instead of metric to avoid memory accumulation in scan
            return (runner_state, update_steps), {}

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            _rng,
        )
        runner_state, _ = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        # Don't accumulate metrics - they're already logged via wandb callback during each update
        return {"runner_state": runner_state}

    return train

def single_run(config):
    config = OmegaConf.to_container(config)
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["AAA", "ADVANTAGE_ALIGNMENT"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f'aaa_cnn_harvest_common'
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    # Enable JIT compilation for massive speedup
    train_jit = jax.jit(make_train(config))

    out = jax.vmap(train_jit)(rngs)

    print("** Saving Results **")
    filename = f'{config["ENV_NAME"]}_seed{config["SEED"]}_reward_{config["REWARD"]}'
    train_state = jax.tree_map(lambda x: x[0], out["runner_state"][0][0][0])
    save_path = f"./checkpoints/{filename}.pkl"
    save_params(train_state, save_path)
    params = load_params(save_path)
    print("** Evaluating Results **")
    evaluate(params, socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]), save_path, config)
    # state_seq = get_rollout(train_state.params, config)
    # viz = OvercookedVisualizer()
    # agent_view_size is hardcoded as it determines the padding around the layout.
    # viz.animate(state_seq, agent_view_size=5, filename=f"{filename}.gif")

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
        # 获取所有智能体的观察
        print(o_t)
        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *env.observation_space()[0].shape)

        # 使用模型选择动作
        network = Actor(action_dim=env.action_space().n, activation=config["ACTIVATION"])  # 使用与训练时相同的参数
        pi= network.apply(params, obs_batch)
        rng, _rng = jax.random.split(rng)
        actions = pi.sample(seed=_rng)
        
        # 转换动作格式
        env_act = {k: v.squeeze() for k, v in unbatchify(
            actions, env.agents, 1, env.num_agents
        ).items()}
        
        # 执行动作
        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(_rng, state, [v.item() for v in env_act.values()])
        done = done["__all__"]
        
        # 记录结果
        # episode_reward += sum(reward.values())
        
        # 渲染
        img = env.render(state)
        pics.append(img)
        
        print('###################')
        print(f'Actions: {env_act}')
        print(f'Reward: {reward}')
        print(f'State: {state.agent_locs}')
        print("###################")
    
    # 保存GIF
    print(f"Saving Episode GIF")
    pics = [Image.fromarray(img) for img in pics]
    pics[0].save(
    f"{root_dir}/state_outer_step_{o_t+1}.gif",
    format="GIF",
    save_all=True,
    optimize=False,
    append_images=pics[1:],
    duration=200,
    loop=0,
    )
        
        # print(f"Episode {episode} total reward: {episode_reward}")
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
        "name": "harvest",
        "method": "grid",
        "metric": {
            "name": "returned_episode_original_returns",
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
        config = copy.deepcopy(default_config)
        # only overwrite the single nested key we're sweeping
        for k, v in dict(wandb.config).items():
            if "." in k:
                parent, child = k.split(".", 1)
                config[parent][child] = v
            else:
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
        # params = load_params(train_state.params)
        # test_env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        # evaluate(params, test_env, config)

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)


@hydra.main(version_base=None, config_path="config", config_name="aaa_cnn_harvest_common")
def main(config):
    if config["TUNE"]:
        tune(config)
    else:
        single_run(config)

if __name__ == "__main__":
    main()