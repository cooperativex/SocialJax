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
        # obs: [batch, num_agents, height, width, channels]
        # actions: [batch, num_agents]
        batch_size, num_agents = obs.shape[:2]
        
        # 重新组织obs为 [batch*num_agents, height, width, channels]
        obs_reshaped = obs.reshape(-1, *obs.shape[2:])
        
        # 分别embed每个agent的观察
        embeddings = CNN(self.activation)(obs_reshaped)  # [batch*num_agents, embed_dim]
        
        # 重新组织为 [batch, num_agents*embed_dim]
        embed_dim = embeddings.shape[-1]
        embeddings = embeddings.reshape(batch_size, num_agents * embed_dim)
        
        # actions one-hot编码
        actions_onehot = nn.one_hot(actions, self.action_dim)  # [batch, num_agents, action_dim]
        actions_onehot = actions_onehot.reshape(batch_size, -1)  # [batch, num_agents*action_dim]
        
        # 拼接所有agents的embeddings和actions
        x = jnp.concatenate([embeddings, actions_onehot], axis=-1)
        
        # 输出集体奖励（标量）[batch]
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        if self.activation == "relu":
            x = nn.relu(x)
        else:
            x = nn.tanh(x)
        x = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        x = jnp.squeeze(x, axis=-1)  # [batch]
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
        
        # 为奖励模型创建单独的优化器（可选）
        reward_lr = config.get("REWARD_LR", config["LR"])  # 如果没有REWARD_LR，使用LR作为默认值
        reward_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(reward_lr, eps=1e-5),
        )
        
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        reward_state = TrainState.create(
            apply_fn=reward_model.apply,
            params=reward_params,
            tx=reward_tx,  # 使用专门的奖励模型优化器
        )
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        
        # 初始化 running statistics 用于 reward model 校准
        # 使用 EMA (Exponential Moving Average) 维护全局统计量
        reward_stats_ema_alpha = config.get("REWARD_STATS_EMA_ALPHA", 0.99)  # EMA 衰减率
        reward_stats = {
            "true_mean": 0.0,
            "true_std": 1.0,
            "pred_mean": 0.0,
            "pred_std": 1.0,
            "count": 0
        }

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            def _env_step(runner_state, unused):
                train_state, reward_state, env_state, last_obs, update_step, rng, reward_mae, reward_stats = runner_state

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
                    """
                    计算 counterfactual regret：如果 agent_id 改变动作，集体奖励的最大改进
                    """
                    def cf_reward_for_action(a_cf):
                        cf_actions = actions.at[:, agent_id].set(a_cf)  # [batch, num_agents]
                        # Reward model 现在直接输出集体奖励 [batch]
                        pred_collective_reward = reward_state.apply_fn(reward_state.params, obs, cf_actions)  # [batch]
                        return pred_collective_reward

                    # 尝试所有可能的动作，计算对应的集体奖励
                    cf_collective_rewards = jax.vmap(cf_reward_for_action)(jnp.arange(num_actions))  # [num_actions, batch]
                    # 当前动作下的集体奖励
                    current_collective_reward = reward_state.apply_fn(reward_state.params, obs, actions)  # [batch]
                    # Regret = 最大可能集体奖励 - 当前集体奖励
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

                # 关键修改：使用 counterfactual reward 而不是 predicted reward
                # Counterfactual reward = 原始奖励 - α * counterfactual regret
                # 或者：使用最优 counterfactual 动作下的奖励
                
                # 获取真实奖励
                true_individual_rewards_batch = batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze()
                true_individual_rewards_reshaped = true_individual_rewards_batch.reshape(-1, env.num_agents)
                true_collective_rewards_batch = jnp.sum(true_individual_rewards_reshaped, axis=1)  # [batch]
                
                # 计算 counterfactual reward
                # 方法1：使用 regret 来调整奖励
                # cf_reward = true_reward - α * regret
                # 方法2：直接使用最优 counterfactual 动作的预测奖励
                # 我们使用方法1，因为它更稳定且考虑了真实奖励
                
                # 获取当前动作下的预测集体奖励（用于计算 regret 的基准）
                pred_collective_rewards_current = reward_state.apply_fn(reward_state.params, obs_for_cf, actions_for_cf)  # [batch]
                
                # 计算每个 agent 的 counterfactual reward
                # 对于每个 agent，counterfactual reward = 真实集体奖励 - regret（该 agent 的遗憾）
                # 注意：cf_regrets 是 [batch, num_agents]，每个 agent 一个 regret
                # 我们需要将 regret 转换为集体奖励的调整
                
                # 方法：使用平均 regret 或最大 regret 来调整集体奖励
                # 或者：为每个 agent 分别计算 counterfactual reward
                cf_regret_mean = jnp.mean(cf_regrets, axis=1)  # [batch] 平均 regret
                cf_regret_max = jnp.max(cf_regrets, axis=1)    # [batch] 最大 regret
                
                # 使用 counterfactual reward = 真实集体奖励 - α * regret
                # α 控制 regret 的影响强度
                cf_alpha = config.get("CF_REWARD_ALPHA", 1.0)  # Counterfactual reward 的权重
                cf_reward_method = config.get("CF_REWARD_METHOD", "mean")  # "mean" 或 "max" 或 "individual"
                
                if cf_reward_method == "mean":
                    # 使用平均 regret
                    cf_collective_rewards = true_collective_rewards_batch - cf_alpha * cf_regret_mean
                elif cf_reward_method == "max":
                    # 使用最大 regret
                    cf_collective_rewards = true_collective_rewards_batch - cf_alpha * cf_regret_max
                else:  # "individual"
                    # 为每个 agent 分别计算（需要 reshape）
                    # 这里我们使用平均作为简化
                    cf_collective_rewards = true_collective_rewards_batch - cf_alpha * cf_regret_mean
                
                # 确保 counterfactual reward 的形状正确
                cf_collective_rewards_expanded = jnp.expand_dims(cf_collective_rewards, axis=1)  # [batch, 1]
                
                shaped_rewards = info["shaped_rewards"].squeeze(-1)
                
                # 动态决定何时开始使用 counterfactual reward：基于 MAE 或固定步数
                reward_switch_start = config.get("REWARD_SWITCH_START", 200)
                reward_switch_end = config.get("REWARD_SWITCH_END", 400)
                reward_mae_threshold_for_switch = config.get("REWARD_MAE_THRESHOLD_FOR_SWITCH", 0.3)
                use_mae_based_switch = config.get("USE_MAE_BASED_SWITCH", True)
                
                # 计算有效的切换开始步数
                use_mae_based_switch_jax = jnp.array(use_mae_based_switch, dtype=jnp.bool_)
                
                def mae_based_switch():
                    mae_ready = reward_mae < reward_mae_threshold_for_switch
                    step_ready = update_step >= reward_switch_start
                    return jnp.where(
                        jnp.logical_and(mae_ready, step_ready),
                        update_step,
                        reward_switch_start
                    )
                
                def fixed_switch():
                    return reward_switch_start
                
                effective_switch_start = jax.lax.cond(
                    use_mae_based_switch_jax,
                    mae_based_switch,
                    fixed_switch
                )
                
                # 计算混合权重：从0平滑增加到1
                safe_switch_end = jnp.maximum(reward_switch_end, effective_switch_start + 1)
                denominator = safe_switch_end - effective_switch_start
                mix_weight = jnp.where(
                    denominator > 1e-6,
                    jnp.clip(
                        (update_step - effective_switch_start) / denominator,
                        0.0, 1.0
                    ),
                    jnp.where(update_step >= effective_switch_start, 1.0, 0.0)
                )
                
                # 确保 counterfactual rewards 有正确的形状（广播到所有 agent）
                cf_collective_rewards_broadcast = jnp.broadcast_to(cf_collective_rewards_expanded, shaped_rewards.shape)
                
                # 渐进式混合：(1-α) * shaped_rewards + α * cf_collective_rewards
                cf_rewards = (1.0 - mix_weight) * shaped_rewards + mix_weight * cf_collective_rewards_broadcast
                
                info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                transition = Transition(
                    batchify_dict(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                    batchify(cf_rewards, env.agents, config["NUM_ACTORS"]).squeeze(),
                    info,
                )
                # 计算当前 reward model 的 MAE（用于动态切换决策）
                # 使用当前 batch 快速估计 MAE
                obs_for_mae = obs_batch.reshape(-1, env.num_agents, *(env.observation_space()[0]).shape)
                actions_for_mae = jnp.reshape(action, (-1, env.num_agents))
                pred_reward_mae = reward_state.apply_fn(reward_state.params, obs_for_mae, actions_for_mae)
                # 计算真实集体奖励
                reward_batched = batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze()
                reward_reshaped = reward_batched.reshape(-1, env.num_agents)
                true_collective = jnp.sum(reward_reshaped, axis=1)
                current_mae = jnp.mean(jnp.abs(pred_reward_mae - true_collective))
                
                runner_state = (train_state, reward_state, env_state, obsv, update_step, rng, current_mae, reward_stats)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, reward_state, env_state, last_obs, update_step, rng, reward_mae, reward_stats = runner_state
            last_obs_batch = jnp.stack([last_obs[:,a,...] for a in env.agents]).reshape(-1, *(env.observation_space()[0]).shape)
            _, last_val = network.apply(train_state.params, last_obs_batch)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value,_ = gae_and_next_value
                    done, value, reward, cf_rewards = (
                        transition.done,
                        transition.value,
                        transition.reward,
                        transition.cf_regret,  # 注意：这里存储的实际上是cf_rewards（混合后的奖励）
                    )

                    # 使用cf_rewards（混合后的奖励）来计算GAE，而不是原始reward
                    # cf_rewards已经包含了shaped_rewards和pred_collective_rewards的混合
                    # 修复：使用更稳定的归一化方式，避免除零和数值不稳定
                    # 不进行归一化，直接使用cf_rewards，但确保数值范围合理
                    # 如果std太小，说明奖励几乎相同，不需要归一化
                    cf_rewards_mean = jnp.mean(cf_rewards, axis=0)
                    cf_rewards_std = jnp.std(cf_rewards, axis=0) + 1e-8
                    # 只在std足够大时才归一化，避免过度归一化
                    should_normalize = cf_rewards_std > 0.1
                    cf_rewards_normalized = jnp.where(
                        should_normalize,
                        (cf_rewards - cf_rewards_mean) / cf_rewards_std,
                        cf_rewards - cf_rewards_mean  # 只中心化，不缩放
                    )
                    
                    # GAE计算：使用处理后的cf_rewards
                    delta = cf_rewards_normalized + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )

                    # 可选：对GAE进行归一化以进一步稳定训练
                    # gae_mean = jnp.mean(gae, axis=0)
                    # gae_std = jnp.std(gae, axis=0) + 1e-8
                    # gae = (gae - gae_mean) / gae_std

                    return (gae, value, cf_rewards), gae

                (_,_,cf_rewards), advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val, last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )


                return (advantages, cf_rewards), advantages + traj_batch.value  # traj.value; value        

            (advantages, cf_rewards), targets = _calculate_gae(traj_batch, last_val)
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
                        action_for_reward = jnp.reshape(traj_batch.action, (-1, env.num_agents))
                        obs_for_reward = jnp.reshape(traj_batch.obs, (-1, env.num_agents, *(traj_batch.obs.shape[1:])))
                        
                        # 获取预测集体奖励 [batch]
                        pred_collective_reward = reward_model.apply(reward_params, obs_for_reward, action_for_reward)
                        
                        # 真实集体奖励：对所有agent的奖励求和
                        true_individual_rewards = jnp.reshape(traj_batch.reward, (-1, env.num_agents))  # [batch, num_agents]
                        true_collective_reward = jnp.sum(true_individual_rewards, axis=1)  # [batch]
                        
                        # 关键修复：归一化预测和真实奖励，确保尺度匹配
                        # 问题：reward model的输出尺度可能和真实奖励不匹配，导致训练不稳定
                        # 解决方案：使用running statistics进行归一化
                        # 计算当前batch的统计量
                        true_mean = jnp.mean(true_collective_reward)
                        true_std = jnp.std(true_collective_reward) + 1e-8
                        pred_mean = jnp.mean(pred_collective_reward)
                        pred_std = jnp.std(pred_collective_reward) + 1e-8
                        
                        # 归一化：将两者都归一化到相同的分布
                        true_normalized = (true_collective_reward - true_mean) / true_std
                        pred_normalized = (pred_collective_reward - pred_mean) / pred_std
                        
                        # 或者：将pred_collective_reward缩放到true_collective_reward的尺度
                        # 这样可以保持真实奖励的原始尺度，只调整预测奖励
                        pred_scaled = (pred_collective_reward - pred_mean) / pred_std * true_std + true_mean
                        
                        # 根据训练阶段调整学习权重（移除步数限制，只基于收敛状态）
                        early_train_boost = config.get("EARLY_TRAIN_BOOST_STEPS", 200)
                        
                        def get_loss_weight():
                            # 只根据早期训练阶段调整权重，不再有步数上限
                            return jax.lax.cond(
                                update_step < early_train_boost,
                                lambda: 3.0,  # 前200步：加强学习权重
                                lambda: 1.5   # 200步后：中等权重（持续训练直到收敛）
                            )
                        
                        loss_weight = get_loss_weight()

                        # 关键修复：使用原始尺度的损失，确保模型学习正确的尺度
                        # 问题：使用归一化损失会导致模型学习归一化后的分布，而不是原始尺度
                        # 解决方案：使用原始尺度的 MSE 损失，让模型直接学习真实奖励的尺度
                        mse_loss_normalized = jnp.mean((pred_normalized - true_normalized) ** 2)
                        mse_loss_scaled = jnp.mean((pred_scaled - true_collective_reward) ** 2)
                        
                        # 使用缩放后的损失（保持原始尺度），这是关键！
                        # 这样模型会学习输出与真实奖励相同尺度的值
                        mse_loss = mse_loss_scaled
                        
                        # 同时添加一个正则项，鼓励预测和真实的均值、方差匹配
                        mean_match_loss = (pred_mean - true_mean) ** 2
                        std_match_loss = (pred_std - true_std) ** 2
                        scale_match_weight = config.get("REWARD_SCALE_MATCH_WEIGHT", 0.1)
                        scale_match_loss = scale_match_weight * (mean_match_loss + std_match_loss)
                        
                        # 总损失 = MSE + 尺度匹配正则项
                        total_mse_loss = mse_loss + scale_match_loss
                        
                        # 计算MAE（用于监控）：使用原始尺度
                        mae = jnp.mean(jnp.abs(pred_collective_reward - true_collective_reward))
                        # 也计算归一化后的MAE
                        mae_normalized = jnp.mean(jnp.abs(pred_normalized - true_normalized))
                        
                        # 详细报告预测和真实奖励之间的差别
                        reward_diff = pred_collective_reward - true_collective_reward
                        reward_diff_mean = jnp.mean(reward_diff)
                        reward_diff_std = jnp.std(reward_diff)
                        reward_diff_max = jnp.max(jnp.abs(reward_diff))
                        
                        # 相对误差（百分比）
                        relative_error = jnp.mean(jnp.abs(reward_diff) / (jnp.abs(true_collective_reward) + 1e-8)) * 100
                        
                        # 相关系数（衡量预测和真实奖励的相关性）
                        pred_centered = pred_collective_reward - pred_mean
                        true_centered = true_collective_reward - true_mean
                        correlation = jnp.sum(pred_centered * true_centered) / (jnp.sqrt(jnp.sum(pred_centered ** 2) * jnp.sum(true_centered ** 2)) + 1e-8)
                        
                        # 分位数误差（中位数绝对误差）
                        median_abs_error = jnp.median(jnp.abs(reward_diff))
                        
                        # 预测是否系统性高估或低估
                        overestimate_ratio = jnp.mean((pred_collective_reward > true_collective_reward).astype(jnp.float32))
                        
                        loss = loss_weight * total_mse_loss
                        return loss, {
                            "reward_loss": loss, 
                            "reward_mae": mae,
                            "reward_mae_normalized": mae_normalized,
                            "pred_reward_mean": jnp.mean(pred_collective_reward),
                            "true_reward_mean": jnp.mean(true_collective_reward),
                            "pred_reward_std": pred_std,
                            "true_reward_std": true_std,
                            "loss_weight": loss_weight,
                            "mse_loss": mse_loss,
                            "mse_loss_normalized": mse_loss_normalized,
                            "mse_loss_scaled": mse_loss_scaled,
                            "scale_match_loss": scale_match_loss,
                            "mean_match_loss": mean_match_loss,
                            "std_match_loss": std_match_loss,
                            "total_mse_loss": total_mse_loss,
                            # 详细差别报告
                            "reward_diff_mean": reward_diff_mean,  # 平均差值（预测-真实）
                            "reward_diff_std": reward_diff_std,    # 差值的标准差
                            "reward_diff_max": reward_diff_max,    # 最大绝对差值
                            "reward_relative_error": relative_error,  # 相对误差（%）
                            "reward_correlation": correlation,     # 相关系数
                            "reward_median_abs_error": median_abs_error,  # 中位数绝对误差
                            "reward_overestimate_ratio": overestimate_ratio,  # 高估比例（预测>真实的比例）
                            "reward_underestimate_ratio": 1.0 - overestimate_ratio  # 低估比例
                        }

                    # 修复：移除策略更新停止机制，让策略持续学习
                    # 之前的策略停止机制会导致策略无法适应环境变化，导致性能崩溃
                    # 如果确实需要限制更新，应该使用学习率衰减而不是完全停止
                    def update_policy(train_state):
                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        total_loss, grads = grad_fn(
                            train_state.params, traj_batch, advantages, targets
                        )
                        train_state = train_state.apply_gradients(grads=grads)
                        return train_state, total_loss
                    
                    # 始终更新策略，不再停止
                    train_state, total_loss = update_policy(train_state)

                    # REWARD MODEL LOSS (移除步数限制，只基于收敛条件停止训练)
                    def train_reward_model():
                        old_reward_params = reward_state.params
                        reward_grad_fn = jax.value_and_grad(reward_loss_fn, has_aux=True)
                        (reward_total_loss, reward_loss_info), reward_grads = reward_grad_fn(
                            reward_state.params, traj_batch
                        )
                        reward_state_updated = reward_state.apply_gradients(grads=reward_grads)
                        
                        # 检查参数是否真的在更新
                        param_change = jax.tree_util.tree_reduce(
                            lambda acc, x: acc + jnp.sum(jnp.abs(x)), 
                            jax.tree_map(lambda old, new: old - new, old_reward_params, reward_state_updated.params),
                            0.0
                        )
                        # 检查梯度大小
                        grad_norm = jax.tree_util.tree_reduce(
                            lambda acc, x: acc + jnp.sum(jnp.abs(x)), 
                            reward_grads,
                            0.0
                        )
                        reward_loss_info["param_change"] = param_change
                        reward_loss_info["grad_norm"] = grad_norm
                        return reward_state_updated, reward_total_loss, reward_loss_info
                    
                    def freeze_reward_model():
                        # 不更新reward model，只计算损失用于监控
                        reward_grad_fn = jax.value_and_grad(reward_loss_fn, has_aux=True)
                        (reward_total_loss, reward_loss_info), _ = reward_grad_fn(
                            reward_state.params, traj_batch
                        )
                        # 参数变化为0，梯度为0
                        reward_loss_info["param_change"] = 0.0
                        reward_loss_info["grad_norm"] = 0.0
                        return reward_state, reward_total_loss, reward_loss_info
                    
                    # 关键修复：防止负反馈循环
                    # 问题：如果reward model在策略使用它的同时继续训练，会形成负反馈循环：
                    # 策略使用reward model预测 → 策略改变 → reward model学习新分布 → 预测改变 → 策略再次改变
                    # 解决方案：一旦开始使用reward model（mix_weight > 0），就停止训练它
                    # 这样可以避免reward model学习到策略变化导致的分布偏移
                    
                    # 检查当前是否在使用reward model（通过检查上一个batch的mix_weight）
                    # 由于我们在_update_minibatch中，需要从外部传入mix_weight信息
                    # 简化方案：基于update_step和MAE来决定是否训练
                    _, temp_loss_info = reward_loss_fn(reward_state.params, traj_batch)
                    reward_mae_threshold = config.get("REWARD_MAE_THRESHOLD", 0.1)
                    reward_converged = temp_loss_info["reward_mae"] < reward_mae_threshold
                    
                    # 关键修复：使用和_env_step中相同的逻辑来计算effective_switch_start
                    # 确保判断"是否开始使用reward model"和实际使用保持一致
                    reward_switch_start = config.get("REWARD_SWITCH_START", 200)
                    reward_switch_end = config.get("REWARD_SWITCH_END", 400)
                    reward_mae_threshold_for_switch = config.get("REWARD_MAE_THRESHOLD_FOR_SWITCH", 0.3)
                    use_mae_based_switch = config.get("USE_MAE_BASED_SWITCH", True)
                    
                    # 使用和_env_step中完全相同的逻辑
                    use_mae_based_switch_jax = jnp.array(use_mae_based_switch, dtype=jnp.bool_)
                    
                    def mae_based_switch():
                        mae_ready = reward_mae < reward_mae_threshold_for_switch
                        step_ready = update_step >= reward_switch_start
                        return jnp.where(
                            jnp.logical_and(mae_ready, step_ready),
                            update_step,
                            reward_switch_start
                        )
                    
                    def fixed_switch():
                        return reward_switch_start
                    
                    effective_switch_start = jax.lax.cond(
                        use_mae_based_switch_jax,
                        mae_based_switch,
                        fixed_switch
                    )
                    
                    # 使用和_env_step中完全相同的逻辑计算mix_weight
                    safe_switch_end = jnp.maximum(reward_switch_end, effective_switch_start + 1)
                    denominator = safe_switch_end - effective_switch_start
                    current_mix_weight = jnp.where(
                        denominator > 1e-6,
                        jnp.clip(
                            (update_step - effective_switch_start) / denominator,
                            0.0, 1.0
                        ),
                        jnp.where(update_step >= effective_switch_start, 1.0, 0.0)
                    )
                    
                    # 关键修复：改变训练策略
                    # 问题：之前的逻辑（mix_weight < 0.01才训练）导致reward model过早停止训练
                    # 新策略：允许reward model在开始使用后继续训练一段时间，直到真正收敛
                    # 这样可以适应策略变化，同时避免长期负反馈循环
                    
                    # 方案1：允许在开始使用后继续训练，但设置一个"稳定期"阈值
                    # 如果mix_weight还很小（< 0.5），说明还在过渡期，继续训练
                    # 如果mix_weight很大（>= 0.5），说明已经主要使用reward model，此时才考虑停止
                    mix_weight_threshold_for_training = config.get("MIX_WEIGHT_THRESHOLD_FOR_TRAINING", 0.8)  # 只有mix_weight超过此值才考虑停止训练
                    
                    # 方案2：设置一个"训练保护期"，在开始使用后的N步内继续训练
                    training_protection_steps = config.get("REWARD_TRAINING_PROTECTION_STEPS", 500)  # 开始使用后继续训练500步
                    steps_since_switch_start = update_step - effective_switch_start
                    in_protection_period = steps_since_switch_start < training_protection_steps
                    
                    # 综合判断：继续训练的条件
                    # 1. 还未收敛 且
                    # 2. (还在保护期内 或 mix_weight还很小)
                    should_train_reward = jnp.logical_and(
                        jnp.logical_not(reward_converged),  # 未收敛
                        jnp.logical_or(
                            in_protection_period,  # 在保护期内
                            current_mix_weight < mix_weight_threshold_for_training  # 或mix_weight还很小
                        )
                    )
                    
                    # 添加监控：记录是否真的在训练
                    def train_reward_model_with_flag():
                        state, loss, info = train_reward_model()
                        info["is_training"] = 1.0
                        return state, loss, info
                    
                    def freeze_reward_model_with_flag():
                        state, loss, info = freeze_reward_model()
                        info["is_training"] = 0.0
                        return state, loss, info
                    
                    reward_state, reward_total_loss, reward_loss_info = jax.lax.cond(
                        should_train_reward,
                        train_reward_model_with_flag,  # 还未使用且未收敛：继续训练
                        freeze_reward_model_with_flag   # 已开始使用或已收敛：冻结训练
                    )
                    


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
            
            metric["reward_loss"] = reward_total_loss.mean()
            metric["total_loss"] = total_loss[0].mean()
            # 添加reward model相关的metrics
            if len(reward_loss_info) > 0:
                # 关键监控：reward model是否真的在训练
                metric["reward_model_is_training"] = reward_loss_info.get("is_training", 0.0).mean()
                metric["reward_mae"] = reward_loss_info["reward_mae"].mean()
                metric["pred_reward_mean"] = reward_loss_info["pred_reward_mean"].mean()
                metric["true_reward_mean"] = reward_loss_info["true_reward_mean"].mean()
                metric["pred_reward_std"] = reward_loss_info.get("pred_reward_std", 0.0).mean()
                metric["true_reward_std"] = reward_loss_info.get("true_reward_std", 0.0).mean()
                metric["reward_param_change"] = reward_loss_info["param_change"].mean()
                metric["reward_grad_norm"] = reward_loss_info["grad_norm"].mean()
                metric["reward_loss_weight"] = reward_loss_info["loss_weight"].mean()
                metric["reward_mse_loss"] = reward_loss_info["mse_loss"].mean()
                # 详细差别报告
                metric["reward_diff_mean"] = reward_loss_info.get("reward_diff_mean", 0.0).mean()  # 平均差值（预测-真实）
                metric["reward_diff_std"] = reward_loss_info.get("reward_diff_std", 0.0).mean()    # 差值标准差
                metric["reward_diff_max"] = reward_loss_info.get("reward_diff_max", 0.0).mean()    # 最大绝对差值
                metric["reward_relative_error"] = reward_loss_info.get("reward_relative_error", 0.0).mean()  # 相对误差（%）
                metric["reward_correlation"] = reward_loss_info.get("reward_correlation", 0.0).mean()     # 相关系数（-1到1，越接近1越好）
                metric["reward_median_abs_error"] = reward_loss_info.get("reward_median_abs_error", 0.0).mean()  # 中位数绝对误差
                metric["reward_overestimate_ratio"] = reward_loss_info.get("reward_overestimate_ratio", 0.5).mean()  # 高估比例（0-1）
                metric["reward_underestimate_ratio"] = reward_loss_info.get("reward_underestimate_ratio", 0.5).mean()  # 低估比例（0-1）
            metric["cf_rewards"] = cf_rewards.mean()
            
            # 添加改进后的监控指标
            reward_switch_start = config.get("REWARD_SWITCH_START", 500)
            reward_switch_end = config.get("REWARD_SWITCH_END", 1000)
            reward_train_end_step = config.get("REWARD_TRAIN_END_STEP", 1200)
            
            # 计算当前混合权重
            current_mix_weight = jnp.clip(
                (update_step - reward_switch_start) / (reward_switch_end - reward_switch_start),
                0.0, 1.0
            )
            
            metric["reward_mix_weight"] = current_mix_weight
            metric["shaped_reward_weight"] = 1.0 - current_mix_weight
            metric["pred_reward_weight"] = current_mix_weight
            metric["reward_model_training"] = jnp.where(update_step < reward_train_end_step, 1.0, 0.0)
            
            # 监控奖励预测精度（如果在训练期间）
            if len(reward_loss_info) > 0:
                metric["reward_prediction_accuracy"] = 1.0 / (1.0 + reward_loss_info["reward_mae"].mean())
                metric["reward_convergence_status"] = jnp.where(
                    reward_loss_info["reward_mae"].mean() < config.get("REWARD_MAE_THRESHOLD", 0.1),
                    1.0, 0.0
                )
            
            # metric["original_rewards"] = metric["original_rewards"].mean() * config["NUM_STEPS"] 
            # metric["shaped_rewards"] = metric["shaped_rewards"].mean() * config["NUM_STEPS"] 
            jax.debug.callback(callback, metric)

            # 更新 reward_mae（从训练循环中获取最新的 MAE）
            current_reward_mae = reward_loss_info["reward_mae"].mean() if len(reward_loss_info) > 0 else 1.0
            
            # 更新 reward_stats（从训练循环中获取最新的统计量）
            # 注意：这里我们需要从 reward_loss_info 中获取统计量，但它在 minibatch 级别
            # 简化方案：保持当前的 reward_stats，它已经在 _env_step 中更新了
            
            runner_state = (train_state, reward_state, env_state, last_obs, update_step, rng, current_reward_mae, reward_stats)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        # 初始化 reward_mae 为一个较大的值，表示 reward model 还未训练
        initial_reward_mae = 10.0
        runner_state = (train_state, reward_state, env_state, obsv, 0, _rng, initial_reward_mae, reward_stats)
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
        name=f'cf_cnn_coin',
        group=f'coin',
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)

    print("** Saving Results **")
    filename = f'{config["ENV_NAME"]}_seed{config["SEED"]}_reward_{config["REWARD"]}'
    train_state = jax.tree_map(lambda x: x[0], out["runner_state"][0])
    reward_state = jax.tree_map(lambda x: x[1], out["runner_state"][0])
    
    # 保存policy和reward model
    save_path_policy = f"./checkpoints/{filename}_policy.pkl"
    save_path_reward = f"./checkpoints/{filename}_reward.pkl"
    save_params(train_state, save_path_policy)
    save_params(reward_state, save_path_reward)
    params = load_params(save_path_policy)
    
    evaluate(params, socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]), save_path_policy, config)
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
        "name": "coin_angle",
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


@hydra.main(version_base=None, config_path="config", config_name="cf_cnn_coin")
def main(config):
    if config["TUNE"]:
        tune(config)
    else:
        single_run(config)

if __name__ == "__main__":
    main()

# --- 使用示例：计算counterfactual regret ---