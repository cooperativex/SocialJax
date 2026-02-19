""" 
Based on PureJaxRL & jaxmarl Implementation of PPO
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import NamedTuple, Any
from flax.training.train_state import TrainState
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

# Import shared MAPPO small network architectures
from algorithms.utils import (
    SmallActor as Actor, SmallCritic as Critic,
    batchify,
    batchify_dict,
    batchify_numpy,
    unbatchify,
    save_params,
    load_params,
    evaluate_mappo_style as evaluate
)



class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    # Individual policy
    ind_action: jnp.ndarray
    ind_value: jnp.ndarray
    ind_log_prob: jnp.ndarray
    # Team policy
    team_action: jnp.ndarray
    team_value: jnp.ndarray
    team_log_prob: jnp.ndarray
    # Rewards
    ind_reward: jnp.ndarray  # Individual reward
    team_reward: jnp.ndarray  # Team reward
    # Observations
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

        # INIT NETWORKS
        # Individual policy and critic (use individual rewards)
        ind_actor_network = Actor(env.action_space().n, activation=config["ACTIVATION"])
        ind_critic_network = Critic(activation=config["ACTIVATION"])

        # Team policy and critic (use team rewards)
        team_actor_network = Actor(env.action_space().n, activation=config["ACTIVATION"])
        team_critic_network = Critic(activation=config["ACTIVATION"])

        rng, _rng_ind_actor, _rng_ind_critic, _rng_team_actor, _rng_team_critic = jax.random.split(rng, 5)

        # Init individual networks (use local observation)
        ac_init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))
        ind_actor_params = ind_actor_network.init(_rng_ind_actor, ac_init_x)
        ind_critic_params = ind_critic_network.init(_rng_ind_critic, ac_init_x)

        # Init team networks (team critic uses global state)
        team_actor_params = team_actor_network.init(_rng_team_actor, ac_init_x)
        obs_shape = env.observation_space()[0].shape
        global_shape = (1, *obs_shape[:-1], obs_shape[-1] * env.num_agents)
        cr_init_x = jnp.zeros(global_shape)
        team_critic_params = team_critic_network.init(_rng_team_critic, cr_init_x)

        # Create optimizers for all 4 networks
        if config["ANNEAL_LR"]:
            ind_actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            ind_critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            team_actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            team_critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            ind_actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            ind_critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            team_actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            team_critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        # Create train states for all 4 networks
        ind_actor_train_state = TrainState.create(
            apply_fn=ind_actor_network.apply,
            params=ind_actor_params,
            tx=ind_actor_tx,
        )
        ind_critic_train_state = TrainState.create(
            apply_fn=ind_critic_network.apply,
            params=ind_critic_params,
            tx=ind_critic_tx,
        )
        team_actor_train_state = TrainState.create(
            apply_fn=team_actor_network.apply,
            params=team_actor_params,
            tx=team_actor_tx,
        )
        team_critic_train_state = TrainState.create(
            apply_fn=team_critic_network.apply,
            params=team_critic_params,
            tx=team_critic_tx,
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

                # Prepare observations
                obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *(env.observation_space()[0]).shape)

                # IRAT: Sample action from team policy and execute it
                # Individual policy evaluates the SAME action for on-policy learning
                rng, rng_team = jax.random.split(rng, 2)

                # Team policy samples action (EXECUTED in environment)
                team_pi = team_actor_network.apply(train_states[2].params, obs_batch)
                team_action = team_pi.sample(seed=rng_team)
                team_log_prob = team_pi.log_prob(team_action)

                # Individual policy evaluates the SAME team_action (on-policy)
                # This allows individual policy to learn from team's trajectory
                # but optimize for individual rewards
                ind_pi = ind_actor_network.apply(train_states[0].params, obs_batch)
                ind_log_prob = ind_pi.log_prob(team_action)  # Use team_action!

                # Execute TEAM action in environment (IRAT design)
                env_act = unbatchify(
                    team_action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = [v for v in env_act.values()]

                # IRAT: Compute values from BOTH critics
                # Individual critic uses local observation
                ind_value = ind_critic_network.apply(train_states[1].params, obs_batch)
                ind_value = ind_value.reshape(config["NUM_ACTORS"])

                # Team critic uses global state
                world_state = jnp.transpose(last_obs, (0,2,3,1,4)).reshape(
                    config["NUM_ENVS"], *(env.observation_space()[0]).shape[:-1], -1
                )
                world_state = jnp.expand_dims(world_state, axis=0)
                world_state = jnp.tile(world_state, (env.num_agents, 1, 1, 1, 1))
                world_state = jnp.reshape(world_state, (-1, *(world_state.shape[2:])))
                team_value = team_critic_network.apply(train_states[3].params, world_state)
                team_value = team_value.reshape(config["NUM_ACTORS"])

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()

                # IRAT: Compute individual and team rewards
                # Individual rewards from environment (NUM_ACTORS,)
                ind_reward = batchify_numpy(reward, env.agents, config["NUM_ACTORS"]).squeeze()

                # Team reward = mean of individual rewards (broadcast to all agents)
                # Using mean instead of sum for better scaling
                # Shape: (NUM_ENVS,) then broadcast to (NUM_ACTORS,)
                ind_reward_reshaped = ind_reward.reshape(env.num_agents, config["NUM_ENVS"])
                team_reward_per_env = ind_reward_reshaped.mean(axis=0)  # (NUM_ENVS,) - CHANGED TO MEAN
                # Broadcast team reward to all agents
                team_reward = jnp.repeat(team_reward_per_env, env.num_agents)  # (NUM_ACTORS,)

                transition = Transition(
                    global_done=jnp.tile(done["__all__"], env.num_agents),
                    done=last_done,
                    ind_action=team_action,  # Individual policy uses same action as team
                    ind_value=ind_value,
                    ind_log_prob=ind_log_prob,
                    team_action=team_action,
                    team_value=team_value,
                    team_log_prob=team_log_prob,
                    ind_reward=ind_reward,
                    team_reward=team_reward,
                    obs=obs_batch,
                    world_state=world_state,
                    info=info
                )
                runner_state = (train_states, env_state, obsv, done_batch, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE (IRAT: compute BOTH individual and team advantages)
            train_states, env_state, last_obs, last_done, rng = runner_state

            # Compute last individual value (uses local observation)
            last_obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *(env.observation_space()[0]).shape)
            last_ind_val = ind_critic_network.apply(train_states[1].params, last_obs_batch)
            last_ind_val = last_ind_val.reshape(config["NUM_ACTORS"])

            # Compute last team value (uses global state)
            last_world_state = jnp.transpose(last_obs, (0,2,3,1,4)).reshape(
                config["NUM_ENVS"], *(env.observation_space()[0]).shape[:-1], -1
            )
            last_world_state = jnp.expand_dims(last_world_state, axis=0)
            last_world_state = jnp.tile(last_world_state, (env.num_agents, 1, 1, 1, 1))
            last_world_state = jnp.reshape(last_world_state, (-1, *(last_world_state.shape[2:])))
            last_team_val = team_critic_network.apply(train_states[3].params, last_world_state)
            last_team_val = last_team_val.reshape(config["NUM_ACTORS"])

            def _calculate_dual_gae(traj_batch, last_ind_val, last_team_val):
                # Individual advantages
                def _get_ind_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.ind_value,
                        transition.ind_reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, ind_advantages = jax.lax.scan(
                    _get_ind_advantages,
                    (jnp.zeros_like(last_ind_val), last_ind_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                ind_targets = ind_advantages + traj_batch.ind_value

                # Team advantages
                def _get_team_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.team_value,
                        transition.team_reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, team_advantages = jax.lax.scan(
                    _get_team_advantages,
                    (jnp.zeros_like(last_team_val), last_team_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                team_targets = team_advantages + traj_batch.team_value

                return ind_advantages, ind_targets, team_advantages, team_targets

            ind_advantages, ind_targets, team_advantages, team_targets = _calculate_dual_gae(
                traj_batch, last_ind_val, last_team_val
            )

            # UPDATE NETWORK (IRAT: 4 networks)
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    ind_actor_ts, ind_critic_ts, team_actor_ts, team_critic_ts = train_states
                    traj_batch, ind_advantages, ind_targets, team_advantages, team_targets = batch_info

                    # --- INDIVIDUAL POLICY LOSS ---
                    def _ind_actor_loss_fn(actor_params, traj_batch, gae):
                        pi = ind_actor_network.apply(
                            actor_params,
                            jnp.reshape(traj_batch.obs, (-1, *(traj_batch.obs).shape[-3:])),
                        )
                        log_prob = pi.log_prob(traj_batch.ind_action.reshape(-1))
                        log_prob = jnp.reshape(log_prob, traj_batch.ind_action.shape)

                        ratio = jnp.exp(log_prob - traj_batch.ind_log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                        entropy = pi.entropy().mean()
                        actor_loss = loss_actor - config["ENT_COEF"] * entropy
                        return actor_loss, (loss_actor, entropy)

                    # --- INDIVIDUAL CRITIC LOSS ---
                    def _ind_critic_loss_fn(critic_params, traj_batch, targets):
                        value = ind_critic_network.apply(
                            critic_params,
                            traj_batch.obs.reshape(-1, *(traj_batch.obs).shape[-3:])
                        )
                        value = jnp.reshape(value, traj_batch.ind_value.shape)

                        value_pred_clipped = traj_batch.ind_value + (value - traj_batch.ind_value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        return config["VF_COEF"] * value_loss, value_loss

                    # --- TEAM POLICY LOSS ---
                    def _team_actor_loss_fn(actor_params, traj_batch, gae):
                        pi = team_actor_network.apply(
                            actor_params,
                            jnp.reshape(traj_batch.obs, (-1, *(traj_batch.obs).shape[-3:])),
                        )
                        log_prob = pi.log_prob(traj_batch.team_action.reshape(-1))
                        log_prob = jnp.reshape(log_prob, traj_batch.team_action.shape)

                        ratio = jnp.exp(log_prob - traj_batch.team_log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                        entropy = pi.entropy().mean()
                        actor_loss = loss_actor - config["ENT_COEF"] * entropy
                        return actor_loss, (loss_actor, entropy)

                    # --- TEAM CRITIC LOSS ---
                    def _team_critic_loss_fn(critic_params, traj_batch, targets):
                        value = team_critic_network.apply(
                            critic_params,
                            traj_batch.world_state.reshape(-1, *(traj_batch.world_state).shape[-3:])
                        )
                        value = jnp.reshape(value, traj_batch.team_value.shape)

                        value_pred_clipped = traj_batch.team_value + (value - traj_batch.team_value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        return config["VF_COEF"] * value_loss, value_loss

                    # Compute gradients and update all 4 networks
                    ind_actor_loss, ind_actor_grads = jax.value_and_grad(_ind_actor_loss_fn, has_aux=True)(
                        ind_actor_ts.params, traj_batch, ind_advantages
                    )
                    ind_critic_loss, ind_critic_grads = jax.value_and_grad(_ind_critic_loss_fn, has_aux=True)(
                        ind_critic_ts.params, traj_batch, ind_targets
                    )
                    team_actor_loss, team_actor_grads = jax.value_and_grad(_team_actor_loss_fn, has_aux=True)(
                        team_actor_ts.params, traj_batch, team_advantages
                    )
                    team_critic_loss, team_critic_grads = jax.value_and_grad(_team_critic_loss_fn, has_aux=True)(
                        team_critic_ts.params, traj_batch, team_targets
                    )

                    # Apply gradients
                    ind_actor_ts = ind_actor_ts.apply_gradients(grads=ind_actor_grads)
                    ind_critic_ts = ind_critic_ts.apply_gradients(grads=ind_critic_grads)
                    team_actor_ts = team_actor_ts.apply_gradients(grads=team_actor_grads)
                    team_critic_ts = team_critic_ts.apply_gradients(grads=team_critic_grads)

                    train_states = (ind_actor_ts, ind_critic_ts, team_actor_ts, team_critic_ts)

                    total_loss = ind_actor_loss[0] + ind_critic_loss[0] + team_actor_loss[0] + team_critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "ind_actor_loss": ind_actor_loss[0],
                        "ind_value_loss": ind_critic_loss[1],
                        "team_actor_loss": team_actor_loss[0],
                        "team_value_loss": team_critic_loss[1],
                        "ind_entropy": ind_actor_loss[1][1],
                        "team_entropy": team_actor_loss[1][1],
                    }

                    return train_states, loss_info

                (
                    train_states,
                    traj_batch,
                    ind_advantages,
                    ind_targets,
                    team_advantages,
                    team_targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"

                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, ind_advantages, ind_targets, team_advantages, team_targets)
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

                train_states, loss_info = jax.lax.scan(
                    _update_minbatch, train_states, minibatches
                )
                update_state = (
                    train_states,
                    traj_batch,
                    ind_advantages,
                    ind_targets,
                    team_advantages,
                    team_targets,
                    rng,
                )
                return update_state, loss_info

            update_state = (
                train_states,
                traj_batch,
                ind_advantages,
                ind_targets,
                team_advantages,
                team_targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_states = update_state[0]
            metric = traj_batch.info
            # loss_info["ratio_0"] = loss_info["ratio"].at[0,0].get()
            # loss_info = jax.tree_map(lambda x: x.mean(), loss_info)
            # metric["loss"] = loss_info
            rng = update_state[-1]

            def callback(metric):
                wandb.log(metric)
            update_steps = update_steps + 1
            metric = jax.tree_map(lambda x: x.mean(), metric)
            metric["update_steps"] = update_steps
            metric["env_step"] = update_steps * config["NUM_STEPS"] * config["NUM_ENVS"]

            # Add loss information to metrics
            loss_metric = jax.tree_map(lambda x: x.mean(), loss_info)
            metric.update(loss_metric)

            # jax.experimental.io_callback(callback, None, metric)

            jax.debug.callback(callback, metric)
            runner_state = (train_states, env_state, last_obs, last_done, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (ind_actor_train_state, ind_critic_train_state, team_actor_train_state, team_critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train

def single_run(config):
    config = OmegaConf.to_container(config)
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IRAT", "INDIVIDUAL_REWARD"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f'irat_cnn_mushrooms'
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    # train_jit = jax.jit(make_train(config))

    train_jit = make_train(config)
    
    out = jax.vmap(train_jit)(rngs)

    print("** Saving Results **")
    filename = f'{config["ENV_NAME"]}_seed{config["SEED"]}_reward_{config["REWARD"]}'
    # IRAT: Save team_actor (index 2) for evaluation
    team_actor_train_state = jax.tree_map(lambda x: x[0], out["runner_state"][0][0][2])
    save_path = f"./checkpoints/{filename}.pkl"
    save_params(team_actor_train_state, save_path)
    params = load_params(save_path)
    print("** Evaluating Results **")
    evaluate(params, socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]), save_path, config)

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
            "SEED": {"values": [42, 52, 62]},
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
        # evaluate_mappo_style(params, test_env, config, use_actor_only=True)

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)

@hydra.main(version_base=None, config_path="config", config_name="IRAT_cnn_mushrooms")
def main(config):
    if config["TUNE"]:
        tune(config)
    else:
        single_run(config)

if __name__ == "__main__":
    main()