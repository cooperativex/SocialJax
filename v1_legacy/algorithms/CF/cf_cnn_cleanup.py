""" 
Based on PureJaxRL & jaxmarl Implementation of PPO
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
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import socialjax
from socialjax.wrappers.baselines import SVOLogWrapper
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

class RewardModel(nn.Module):
    agent_number: int
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs, actions):
        # obs: [batch, ...]
        # actions: [batch, num_agents] 或 [num_agents]
        embedding = CNN(self.activation)(obs)
        # actions one-hot编码
        actions_onehot = nn.one_hot(actions, self.action_dim)  # [batch, num_agents, action_dim]
        actions_onehot = actions_onehot.reshape((embedding.shape[0], -1))  # [batch, num_agents*action_dim]
        x = jnp.concatenate([embedding, actions_onehot], axis=-1)
        x = nn.Dense(self.agent_number, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
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
    cf_regret: jnp.ndarray
    info: jnp.ndarray


def get_rollout(params, config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
    reward_model = RewardModel(env.num_agents, env.action_space().n)
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

    env = SVOLogWrapper(env, replace_info=False)

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
        reward_model = RewardModel(env.num_agents, env.action_space().n)
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))
        init_reward_x = jnp.zeros((1, env.num_agents, *(env.observation_space()[0]).shape))
        init_actions = jnp.zeros((1, env.num_agents), dtype=jnp.int32)  # 假设动作空间是int类型

        network_params = network.init(_rng, init_x)
        reward_params = reward_model.init(_rng, init_reward_x, init_actions)
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
        reward_state = TrainState.create(
            apply_fn=reward_model.apply,
            params=reward_params,
            tx=tx,
        )
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            def _env_step(runner_state, unused):
                train_state, reward_state, env_state, last_obs, update_step, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)


                obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *(env.observation_space()[0]).shape)
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

                
                def compute_cf_regret(obs, actions, reward_state, agent_id, num_actions):
                    num_agents = actions.shape[1]
                    weights = 1.0 - jax.nn.one_hot(agent_id, num_agents)  # [num_agents]

                    def cf_reward_for_action(a_cf):
                        cf_actions = actions.at[:, agent_id].set(a_cf)  # [batch, num_agents]
                        pred_rewards = reward_state.apply_fn(reward_state.params, obs, cf_actions)  # [batch, num_agents]
                        collective_reward = jnp.sum(pred_rewards * weights, axis=1)  # [batch]
                        return collective_reward

                    cf_collective_rewards = jax.vmap(cf_reward_for_action)(jnp.arange(num_actions))  # [num_actions, batch]
                    current_pred_rewards = reward_state.apply_fn(reward_state.params, obs, actions)  # [batch, num_agents]
                    current_collective_reward = jnp.sum(current_pred_rewards * weights, axis=1)  # [batch]
                    regret = jnp.max(cf_collective_rewards, axis=0) - current_collective_reward  # [batch]
                    return regret


                # --- counterfactual regret ---
                num_agents = env.num_agents
                num_actions = env.action_space().n
                actions_for_cf = jnp.reshape(action, (-1, num_agents))  # [batch, num_agents]
                obs_for_cf = obs_batch.reshape(-1, env.num_agents, *(env.observation_space()[0]).shape)
                cf_regrets = jax.vmap(
                    lambda agent_id: compute_cf_regret(obs_for_cf, actions_for_cf, reward_state, agent_id, num_actions)
                )(jnp.arange(num_agents))  # [num_agents, batch]
                cf_regrets = jnp.transpose(cf_regrets, (1, 0))  # [batch, num_agents]

                info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                transition = Transition(
                    batchify_dict(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                    batchify(cf_regrets, env.agents, config["NUM_ACTORS"]).squeeze(),
                    info,
                )
                runner_state = (train_state, reward_state, env_state, obsv, update_step, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, reward_state, env_state, last_obs, update_step, rng = runner_state
            last_obs_batch = jnp.stack([last_obs[:,a,...] for a in env.agents]).reshape(-1, *(env.observation_space()[0]).shape)
            _, last_val = network.apply(train_state.params, last_obs_batch)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward, cf_regret = (
                        transition.done,
                        transition.value,
                        transition.reward,
                        transition.cf_regret,
                    )

                    reward_mean = jnp.mean(reward, axis=0)
                    # reward_std = jnp.std(reward, axis=0) + 1e-8
                    reward = (reward - reward_mean)# / reward_std
                    cf_regret = (cf_regret - cf_regret.mean()) / (cf_regret.std() + 1e-8)
                    shaped_reward = reward - config["REWARD_ALPHA"] * cf_regret

                    delta = shaped_reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )

                    # gae_mean = jnp.mean(gae)
                    # gae_std = jnp.std(gae, axis=0) + 1e-8
                    # gae = (gae - gae_mean) / gae_std

                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )


                return advantages, advantages + traj_batch.value  # traj.value; value
            
            

            advantages, targets = _calculate_gae(traj_batch, last_val)
            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                train_state, reward_state, traj_batch, advantages, targets, rng, update_step = update_state
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

                # UPDATE NETWORK
                def _update_minibatch(states, batch_info):
                    train_state, reward_state = states
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

                    def reward_loss_fn(reward_params, traj_batch):
                        obs_for_reward = jnp.reshape(traj_batch.obs, (-1, env.num_agents, *traj_batch.obs.shape[2:]))
                        action_for_reward = jnp.reshape(traj_batch.action, (-1, env.num_agents))
                        
                        pred_reward = reward_model.apply(reward_params, obs_for_reward, action_for_reward)
                        pred_reward = jnp.reshape(pred_reward, (-1, env.num_agents))
                        
                        true_reward = jnp.reshape(traj_batch.reward, (-1, env.num_agents))

                        loss = jnp.mean((pred_reward - true_reward) ** 2)
                        return loss, {"reward_loss": loss}

                    # 条件更新策略网络：只有在指定步数之前才更新
                    policy_stop_step = config.get("POLICY_STOP_STEP", 1000)  # 默认1000步后停止策略更新
                    
                    def update_policy():
                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        total_loss, grads = grad_fn(
                            train_state.params, traj_batch, advantages, targets
                        )
                        train_state = train_state.apply_gradients(grads=grads)
                        return train_state, total_loss
                    
                    def skip_policy_update():
                        total_loss = (0, (0, 0, 0))  # Dummy loss
                        return train_state, total_loss
                    
                    train_state, total_loss = jax.lax.cond(
                        update_step < policy_stop_step,
                        update_policy,
                        skip_policy_update
                    )

                    # REWARD MODEL LOSS (始终更新)
                    reward_grad_fn = jax.value_and_grad(reward_loss_fn, has_aux=True)
                    (reward_total_loss, reward_loss_info), reward_grads = reward_grad_fn(
                        reward_state.params, traj_batch
                    )
                    reward_state = reward_state.apply_gradients(grads=reward_grads)

                    return (train_state, reward_state), (total_loss, reward_total_loss, reward_loss_info)
                
                (train_state, reward_state), loss_info = jax.lax.scan(
                    _update_minibatch, (train_state, reward_state), minibatches
                )
                return (train_state, reward_state, traj_batch, advantages, targets, rng, update_step), loss_info

            update_state = (train_state, reward_state, traj_batch, advantages, targets, rng, update_step)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, (), config["UPDATE_EPOCHS"]
            )
            
            total_loss, reward_total_loss, reward_loss_info = loss_info
            train_state, reward_state = update_state[0], update_state[1]
            metric = traj_batch.info
            rng = update_state[-2]  # 注意：现在update_step在倒数第二个位置
            update_step = update_state[-1]

            def callback(metric):
                wandb.log(metric)

            update_step = update_step + 1
            metric = jax.tree_map(lambda x: x.mean(), metric)
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            metric["advantages"] = advantages.mean()
            metric["clean_action_info"] = metric["clean_action_info"] * config["ENV_KWARGS"]["num_inner_steps"]
            metric["reward_loss"] = reward_total_loss.mean()
            metric["total_loss"] = total_loss[0].mean()
            # metric["original_rewards"] = metric["original_rewards"].mean() * config["NUM_STEPS"] 
            # metric["shaped_rewards"] = metric["shaped_rewards"].mean() * config["NUM_STEPS"] 
            jax.debug.callback(callback, metric)

            runner_state = (train_state, reward_state, env_state, last_obs, update_step, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, reward_state, env_state, obsv, 0, _rng)
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
        tags=["CF", "CNN"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f'cf_cnn_cleanup',
        group=f'cleanup',
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)

    print("** Saving Results **")
    filename = f'{config["ENV_NAME"]}_seed{config["SEED"]}_reward_{config["REWARD"]}'
    train_state = jax.tree_map(lambda x: x[0], out["runner_state"][0])
    save_path = f"./checkpoints/{filename}.pkl"
    save_params(train_state, save_path)
    params = load_params(save_path)
    
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
    root_dir = f"evaluation/cleanup"
    path = Path(root_dir + "/state_pics")
    path.mkdir(parents=True, exist_ok=True)

    for o_t in range(config["GIF_NUM_FRAMES"]):
        # 获取所有智能体的观察
        print(o_t)
        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *env.observation_space()[0].shape)

        # 使用模型选择动作
        network = ActorCritic(action_dim=env.action_space().n, activation=config["ACTIVATION"])  # 使用与训练时相同的参数
        pi, _ = network.apply(params, obs_batch)
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
        "name": "cleanup_angle",
        "method": "grid",
        "metric": {
            "name": "returned_episode_original_returns",
            "goal": "maximize",
        },
        "parameters": {
            # "LR": {"values": [0.001, 0.0005, 0.0001, 0.00005]},
            # "ACTIVATION": {"values": ["relu", "tanh"]},
            # "UPDATE_EPOCHS": {"values": [2, 4, 8]},
            # "NUM_MINIBATCHES": {"values": [4, 8, 16, 32]},
            # "CLIP_EPS": {"values": [0.1, 0.2, 0.3]},
            # "ENT_COEF": {"values": [0.001, 0.01, 0.1]},
            # "NUM_STEPS": {"values": [64, 128, 256]},
            "ENV_KWARGS.svo_w": {"values": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]},
            # "ENV_KWARGS.svo_ideal_angle_degrees": {"values": [0, 45, 90]},

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


@hydra.main(version_base=None, config_path="config", config_name="cf_cnn_cleanup")
def main(config):
    if config["TUNE"]:
        tune(config)
    else:
        single_run(config)

if __name__ == "__main__":
    main()

# --- 使用示例：计算counterfactual regret ---