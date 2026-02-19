"""
示例：如何将因果Reward Model集成到现有代码中
"""

import jax
import jax.numpy as jnp
from causal_reward_model import CausalRewardModel, generate_counterfactual_samples_jax, causal_consistency_loss


def reward_loss_fn_with_causal(reward_params, traj_batch, reward_model, config, update_step, rng):
    """
    带因果损失的reward loss函数
    
    这是对原有reward_loss_fn的扩展，添加了因果一致性损失
    """
    action_for_reward = jnp.reshape(traj_batch.action, (-1, env.num_agents))
    obs_for_reward = jnp.reshape(traj_batch.obs, (-1, env.num_agents, *(traj_batch.obs.shape[1:])))
    
    # 1. 基础预测
    pred_individual_rewards = reward_model.apply(reward_params, obs_for_reward, action_for_reward)
    true_individual_rewards = jnp.reshape(traj_batch.reward, (-1, env.num_agents))
    
    # 2. 计算基础损失（MSE或Huber）
    mse_loss_individual = jnp.mean((pred_individual_rewards - true_individual_rewards) ** 2)
    pred_collective_reward = jnp.sum(pred_individual_rewards, axis=1)
    true_collective_reward = jnp.sum(true_individual_rewards, axis=1)
    mse_loss_collective = jnp.mean((pred_collective_reward - true_collective_reward) ** 2)
    
    # 3. 生成反事实样本（如果启用）
    use_causal_training = config.get("USE_CAUSAL_TRAINING", False)
    causal_loss = 0.0
    
    if use_causal_training:
        rng, rng_cf = jax.random.split(rng)
        num_cf_samples = config.get("CF_NUM_SAMPLES", 5)
        
        # 定义reward model函数
        def reward_model_fn(obs, actions):
            return reward_model.apply(reward_params, obs, actions)
        
        # 生成反事实样本
        cf_actions_list, cf_pred_rewards_list = generate_counterfactual_samples_jax(
            obs_for_reward, action_for_reward, reward_model_fn, rng_cf, num_cf_samples
        )
        
        # 计算因果一致性损失
        # 注意：这里我们没有真实的反事实奖励，所以使用预测的变化的方差作为正则项
        consistency_weight = config.get("CAUSAL_CONSISTENCY_WEIGHT", 0.1)
        causal_loss = causal_consistency_loss(
            pred_individual_rewards, 
            cf_pred_rewards_list,  # [num_samples, batch, num_agents]
            true_individual_rewards,
            None  # 没有真实反事实奖励
        )
        causal_loss = consistency_weight * causal_loss
    
    # 4. 总损失
    collective_loss_weight = config.get("COLLECTIVE_LOSS_WEIGHT", 0.5)
    total_loss = mse_loss_individual + collective_loss_weight * mse_loss_collective + causal_loss
    
    # 5. 返回损失和监控信息
    mae = jnp.mean(jnp.abs(pred_collective_reward - true_collective_reward))
    
    return total_loss, {
        "reward_loss": total_loss,
        "reward_mae": mae,
        "mse_loss_individual": mse_loss_individual,
        "mse_loss_collective": mse_loss_collective,
        "causal_loss": causal_loss,
        "pred_reward_mean": jnp.mean(pred_collective_reward),
        "true_reward_mean": jnp.mean(true_collective_reward),
    }


# 使用示例：在make_train函数中替换RewardModel
def example_integration():
    """
    集成示例：如何在主训练循环中使用CausalRewardModel
    """
    # 原来的代码：
    # reward_model = RewardModel(env.num_agents, env.action_space().n)
    
    # 改为：
    reward_model = CausalRewardModel(
        agent_number=env.num_agents,
        action_dim=env.action_space().n,
        use_causal_attention=True,  # 启用因果注意力
        num_attention_heads=4,  # 注意力头数
        activation="relu"
    )
    
    # 初始化参数（需要传入causal_mask，如果使用）
    causal_mask = None  # 或者定义agent之间的因果关系
    # 例如：agent 0 影响 agent 1
    # causal_mask = jnp.array([
    #     [0, 0, 0],  # agent 0 不影响其他
    #     [1, 0, 0],  # agent 1 影响 agent 0
    #     [1, 1, 0],  # agent 2 影响 agent 0 和 1
    # ])
    
    init_obs = jnp.zeros((1, env.num_agents, *(env.observation_space()[0]).shape))
    init_actions = jnp.zeros((1, env.num_agents), dtype=jnp.int32)
    reward_params = reward_model.init(
        rng, init_obs, init_actions, causal_mask=causal_mask
    )
    
    # 在训练时使用
    pred_rewards = reward_model.apply(
        reward_params, obs, actions, causal_mask=causal_mask
    )

