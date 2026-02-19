"""
Causal Reward Model with Causal Attention Mechanism
使用因果方法改进reward prediction：
1. 因果注意力：建模agent之间的因果关系
2. 反事实推理：使用反事实样本增强训练
3. 因果特征分解：区分直接和间接因果效应
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from typing import Sequence
import numpy as np


class CausalAttention(nn.Module):
    """因果注意力机制：建模agent之间的因果关系"""
    num_heads: int = 4
    head_dim: int = 16
    
    @nn.compact
    def __call__(self, embeddings, actions, mask=None):
        """
        embeddings: [batch, num_agents, embed_dim]
        actions: [batch, num_agents]
        mask: [batch, num_agents, num_agents] - 因果掩码（可选）
        """
        batch_size, num_agents, embed_dim = embeddings.shape
        
        # 将actions编码为特征
        action_dim = actions.shape[-1] if len(actions.shape) > 2 else 1
        if len(actions.shape) == 2:
            # actions是整数，需要one-hot
            actions_onehot = nn.one_hot(actions, action_dim)  # [batch, num_agents, action_dim]
        else:
            actions_onehot = actions
        
        # 将actions特征与embeddings结合
        action_embed = nn.Dense(embed_dim)(actions_onehot)  # [batch, num_agents, embed_dim]
        x = embeddings + action_embed  # [batch, num_agents, embed_dim]
        
        # Multi-head attention
        q = nn.Dense(self.num_heads * self.head_dim)(x)  # [batch, num_agents, num_heads * head_dim]
        k = nn.Dense(self.num_heads * self.head_dim)(x)
        v = nn.Dense(self.num_heads * self.head_dim)(x)
        
        q = q.reshape(batch_size, num_agents, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, num_agents, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, num_agents, self.num_heads, self.head_dim)
        
        # 计算注意力分数
        scores = jnp.einsum('bqhd,bkhd->bhqk', q, k) / jnp.sqrt(self.head_dim)  # [batch, num_heads, num_agents, num_agents]
        
        # 应用因果掩码（如果提供）
        # 因果掩码确保agent i只能关注到agent j（如果j对i有因果影响）
        if mask is not None:
            scores = scores + mask * (-1e9)  # 将masked位置设为负无穷
        
        # Softmax
        attn_weights = nn.softmax(scores, axis=-1)  # [batch, num_heads, num_agents, num_agents]
        
        # 应用注意力
        out = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, v)  # [batch, num_agents, num_heads, head_dim]
        out = out.reshape(batch_size, num_agents, self.num_heads * self.head_dim)
        
        # 输出投影
        out = nn.Dense(embed_dim)(out)  # [batch, num_agents, embed_dim]
        
        return out, attn_weights


class CausalRewardModel(nn.Module):
    """基于因果方法的Reward Model"""
    agent_number: int
    action_dim: int
    activation: str = "relu"
    use_causal_attention: bool = True
    num_attention_heads: int = 4
    
    @nn.compact
    def __call__(self, obs, actions, causal_mask=None):
        """
        obs: [batch, num_agents, height, width, channels]
        actions: [batch, num_agents]
        causal_mask: [batch, num_agents, num_agents] - 可选的因果掩码
        """
        batch_size, num_agents = obs.shape[:2]
        
        # 1. 提取每个agent的观察特征
        obs_reshaped = obs.reshape(-1, *obs.shape[2:])  # [batch*num_agents, height, width, channels]
        
        # 使用CNN提取特征
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        
        # CNN特征提取
        x = nn.Conv(
            features=32,
            kernel_size=(5, 5),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs_reshaped)
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
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = activation(x)
        
        # 重新组织为 [batch, num_agents, embed_dim]
        embed_dim = x.shape[-1]
        embeddings = x.reshape(batch_size, num_agents, embed_dim)
        
        # 2. 因果注意力机制：建模agent之间的因果关系
        if self.use_causal_attention:
            causal_attn = CausalAttention(
                num_heads=self.num_attention_heads,
                head_dim=embed_dim // self.num_attention_heads
            )
            causal_embeddings, attn_weights = causal_attn(embeddings, actions, causal_mask)
            # 残差连接
            embeddings = embeddings + causal_embeddings
        
        # 3. 因果特征分解：区分直接效应和间接效应
        # 直接效应：agent自己的观察和动作
        direct_effects = embeddings  # [batch, num_agents, embed_dim]
        
        # 间接效应：通过其他agent的观察和动作
        if self.use_causal_attention:
            # 使用注意力权重来加权其他agent的影响
            # attn_weights: [batch, num_heads, num_agents, num_agents]
            # 平均所有head的注意力权重
            avg_attn = jnp.mean(attn_weights, axis=1)  # [batch, num_agents, num_agents]
            # 移除自注意力（对角线）
            mask_self = jnp.eye(num_agents)[None, :, :]  # [1, num_agents, num_agents]
            indirect_attn = avg_attn * (1 - mask_self)  # [batch, num_agents, num_agents]
            # 计算间接效应
            indirect_effects = jnp.einsum('bij,bjd->bid', indirect_attn, embeddings)  # [batch, num_agents, embed_dim]
        else:
            # 如果没有注意力，使用简单的平均池化
            indirect_effects = jnp.mean(embeddings, axis=1, keepdims=True)  # [batch, 1, embed_dim]
            indirect_effects = jnp.broadcast_to(indirect_effects, embeddings.shape)
        
        # 4. 组合直接和间接效应
        # 使用可学习的权重来平衡直接和间接效应
        # 使用sigmoid确保权重在合理范围内
        direct_weight_raw = self.param('direct_weight', 
                                      lambda rng, shape: jnp.ones(shape) * 0.7, 
                                      (1,))
        indirect_weight_raw = self.param('indirect_weight',
                                         lambda rng, shape: jnp.ones(shape) * 0.3,
                                         (1,))
        
        # 归一化权重
        total_weight = direct_weight_raw + indirect_weight_raw + 1e-8
        direct_weight = direct_weight_raw / total_weight
        indirect_weight = indirect_weight_raw / total_weight
        
        combined_features = (direct_weight * direct_effects + 
                            indirect_weight * indirect_effects)  # [batch, num_agents, embed_dim]
        
        # 5. 预测每个agent的奖励
        # 将特征展平
        combined_flat = combined_features.reshape(batch_size, -1)  # [batch, num_agents * embed_dim]
        
        # 输出层
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(combined_flat)
        x = activation(x)
        x = nn.Dense(num_agents, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)  # [batch, num_agents]
        
        return x


def generate_counterfactual_samples_jax(obs, actions, reward_model_fn, rng, num_samples=5, num_actions=None):
    """
    生成反事实样本用于数据增强（JAX版本）
    
    对于每个样本，随机改变一个agent的动作，预测奖励
    这样可以学习到"如果某个agent改变动作，奖励会如何变化"
    
    Args:
        obs: 观察 [batch, num_agents, ...]
        actions: 动作 [batch, num_agents] (整数) 或 [batch, num_agents, action_dim] (one-hot)
        reward_model_fn: reward model函数
        rng: JAX随机数生成器
        num_samples: 反事实样本数量
        num_actions: 动作空间大小（如果actions是整数，必须提供）
    """
    batch_size, num_agents = actions.shape[:2]
    
    # 确定动作空间大小
    if num_actions is None:
        if len(actions.shape) == 2:
            # actions是整数，需要从外部传入num_actions
            raise ValueError("num_actions must be provided when actions are integers")
        else:
            # actions是one-hot
            num_actions = actions.shape[-1]
    else:
        # 确保num_actions是JAX数组
        if not isinstance(num_actions, (jnp.ndarray, int)):
            num_actions = jnp.array(num_actions, dtype=jnp.int32)
    
    # 确保num_actions是Python int（静态值）
    if isinstance(num_actions, jnp.ndarray):
        num_actions = int(num_actions)
    elif not isinstance(num_actions, int):
        num_actions = int(num_actions)
    
    def generate_one_cf(rng):
        rng_agent, rng_action = jax.random.split(rng)
        # 随机选择一个agent
        agent_id = jax.random.randint(rng_agent, (), 0, num_agents)
        # 随机选择一个不同的动作
        if len(actions.shape) == 2:
            # actions是整数
            current_action = actions[:, agent_id]
            # 使用JAX兼容的方式生成新动作（num_actions必须是Python int）
            new_action = jax.random.randint(rng_action, current_action.shape, 0, num_actions)
            # 确保新动作与当前动作不同
            new_action = jnp.where(new_action == current_action, 
                                  (new_action + 1) % num_actions, 
                                  new_action)
            cf_actions = actions.at[:, agent_id].set(new_action)
        else:
            # actions是one-hot，需要转换
            cf_actions = actions.copy()
            # 随机改变动作（简化：循环移位）
            cf_actions = cf_actions.at[:, agent_id].set(
                jnp.roll(cf_actions[:, agent_id], 1, axis=-1)
            )
        
        # 预测反事实奖励
        cf_reward = reward_model_fn(obs, cf_actions)
        return cf_actions, cf_reward
    
    # 生成多个反事实样本
    rngs = jax.random.split(rng, num_samples)
    cf_actions_list, cf_rewards_list = jax.vmap(generate_one_cf)(rngs)
    
    return cf_actions_list, cf_rewards_list


def causal_consistency_loss(pred_rewards, cf_pred_rewards, true_rewards, cf_true_rewards):
    """
    因果一致性损失：确保反事实预测的一致性
    
    如果agent i改变动作，应该能够预测到奖励的变化
    
    Args:
        pred_rewards: [batch, num_agents] - 原始预测奖励
        cf_pred_rewards: [num_samples, batch, num_agents] - 反事实预测奖励
        true_rewards: [batch, num_agents] - 原始真实奖励
        cf_true_rewards: [num_samples, batch, num_agents] - 反事实真实奖励（如果可用）
    """
    # 计算奖励变化
    # 对于每个反事实样本，计算预测和真实的变化
    reward_change_pred = cf_pred_rewards - pred_rewards[None, :, :]  # [num_samples, batch, num_agents]
    
    if cf_true_rewards is not None:
        reward_change_true = cf_true_rewards - true_rewards[None, :, :]
        # 一致性损失：预测的变化应该与真实变化一致
        consistency_loss = jnp.mean((reward_change_pred - reward_change_true) ** 2)
    else:
        # 如果没有真实反事实奖励，使用预测的变化的方差作为正则项
        # 鼓励预测的变化是合理的（不应该太大或太小）
        reward_change_mean = jnp.mean(reward_change_pred, axis=0)  # [batch, num_agents]
        reward_change_var = jnp.var(reward_change_pred, axis=0)  # [batch, num_agents]
        # 鼓励变化的一致性（方差不应该太大）
        consistency_loss = jnp.mean(reward_change_var)
    
    return consistency_loss


def causal_intervention_loss(pred_rewards, true_rewards, intervention_mask):
    """
    因果干预损失：确保模型能够正确预测干预效果
    
    intervention_mask: [batch, num_agents] - 指示哪些agent被干预
    """
    # 对于被干预的agent，预测应该更准确
    intervention_weight = intervention_mask.astype(jnp.float32)
    
    # 加权MSE损失
    mse = (pred_rewards - true_rewards) ** 2
    weighted_mse = jnp.mean(mse * intervention_weight[:, :, None] if len(intervention_weight.shape) == 2 else mse * intervention_weight)
    
    return weighted_mse

