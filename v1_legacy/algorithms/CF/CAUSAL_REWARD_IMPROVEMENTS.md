# 使用因果方法改进Reward Prediction

## 问题分析

当前的reward model存在以下问题：
1. **忽略因果关系**：简单拼接所有agent的特征，没有建模agent之间的因果关系
2. **混淆变量**：可能学习到虚假的相关性而非真实的因果关系
3. **缺乏反事实推理**：无法回答"如果某个agent改变动作，奖励会如何变化"

## 因果方法改进方案

### 1. 因果注意力机制（Causal Attention）

**核心思想**：使用注意力机制建模agent之间的因果关系

```python
class CausalAttention(nn.Module):
    """因果注意力机制：建模agent之间的因果关系"""
    num_heads: int = 4
    head_dim: int = 16
```

**优势**：
- 学习agent之间的因果关系强度
- 可以应用因果掩码（causal mask）来约束注意力
- 通过注意力权重可视化因果关系

### 2. 因果特征分解（Causal Feature Decomposition）

**核心思想**：将奖励分解为直接效应和间接效应

- **直接效应**：agent自己的观察和动作对奖励的影响
- **间接效应**：通过其他agent的观察和动作对奖励的影响

```python
# 直接效应：agent自己的观察和动作
direct_effects = embeddings

# 间接效应：通过其他agent的影响
indirect_effects = attention_weighted_other_agents

# 组合
combined = direct_weight * direct_effects + indirect_weight * indirect_effects
```

**优势**：
- 明确区分直接和间接因果效应
- 可学习的权重平衡两种效应
- 提高可解释性

### 3. 反事实数据增强（Counterfactual Data Augmentation）

**核心思想**：生成反事实样本，学习"如果改变动作，奖励会如何变化"

```python
def generate_counterfactual_samples(obs, actions, reward_model, num_samples=5):
    """生成反事实样本用于数据增强"""
    # 随机改变一个agent的动作
    # 预测反事实奖励
    # 学习奖励变化模式
```

**优势**：
- 增加训练数据多样性
- 学习反事实推理能力
- 提高泛化能力

### 4. 因果一致性损失（Causal Consistency Loss）

**核心思想**：确保反事实预测的一致性

```python
def causal_consistency_loss(pred_rewards, cf_pred_rewards, true_rewards, cf_true_rewards):
    """因果一致性损失：确保反事实预测的一致性"""
    reward_change_pred = cf_pred_rewards - pred_rewards
    reward_change_true = cf_true_rewards - true_rewards
    consistency_loss = mean((reward_change_pred - reward_change_true) ** 2)
```

**优势**：
- 确保模型能够正确预测干预效果
- 提高反事实推理的准确性
- 减少虚假相关性

### 5. 因果干预损失（Causal Intervention Loss）

**核心思想**：对干预的agent给予更高的权重

```python
def causal_intervention_loss(pred_rewards, true_rewards, intervention_mask):
    """因果干预损失：确保模型能够正确预测干预效果"""
    # 对于被干预的agent，预测应该更准确
    weighted_mse = mean(mse * intervention_weight)
```

## 集成方案

### 步骤1：替换RewardModel

```python
# 原来的
reward_model = RewardModel(env.num_agents, env.action_space().n)

# 改为
from causal_reward_model import CausalRewardModel
reward_model = CausalRewardModel(
    agent_number=env.num_agents,
    action_dim=env.action_space().n,
    use_causal_attention=True,
    num_attention_heads=4
)
```

### 步骤2：修改损失函数

在`reward_loss_fn`中添加因果损失：

```python
def reward_loss_fn(reward_params, traj_batch):
    # ... 原有的损失计算 ...
    
    # 添加反事实样本生成
    cf_samples = generate_counterfactual_samples(
        obs_for_reward, action_for_reward, 
        lambda obs, acts: reward_model.apply(reward_params, obs, acts),
        num_samples=config.get("CF_NUM_SAMPLES", 5)
    )
    
    # 计算反事实预测
    cf_pred_rewards = []
    cf_true_rewards = []
    for cf_obs, cf_actions, _ in cf_samples:
        cf_pred = reward_model.apply(reward_params, cf_obs, cf_actions)
        # 获取反事实真实奖励（需要环境交互，或使用近似）
        cf_pred_rewards.append(cf_pred)
    
    # 添加因果一致性损失
    consistency_weight = config.get("CAUSAL_CONSISTENCY_WEIGHT", 0.1)
    consistency_loss = causal_consistency_loss(
        pred_individual_rewards, cf_pred_rewards,
        true_individual_rewards, cf_true_rewards
    )
    
    # 总损失
    total_loss = mse_loss + consistency_weight * consistency_loss
```

### 步骤3：添加因果掩码（可选）

如果知道agent之间的因果关系结构，可以添加因果掩码：

```python
# 例如：agent 0 影响 agent 1，但不影响 agent 2
causal_mask = jnp.array([
    [0, 0, 0],  # agent 0 不影响任何其他agent
    [1, 0, 0],  # agent 1 只影响 agent 0
    [1, 1, 0],  # agent 2 影响 agent 0 和 1
])

# 在调用reward model时传入
pred_rewards = reward_model.apply(
    reward_params, obs, actions, 
    causal_mask=causal_mask
)
```

## 预期效果

1. **提高预测准确率**：
   - 因果注意力机制更好地建模agent间关系
   - 因果特征分解减少混淆变量影响

2. **提高泛化能力**：
   - 反事实数据增强增加数据多样性
   - 因果一致性损失确保反事实推理正确

3. **提高可解释性**：
   - 注意力权重可视化因果关系
   - 直接/间接效应分解提供解释

## 配置参数

```yaml
# 因果方法相关配置
USE_CAUSAL_ATTENTION: true  # 是否使用因果注意力
NUM_ATTENTION_HEADS: 4  # 注意力头数
CAUSAL_CONSISTENCY_WEIGHT: 0.1  # 因果一致性损失权重
CF_NUM_SAMPLES: 5  # 反事实样本数量
USE_CAUSAL_MASK: false  # 是否使用预定义的因果掩码
```

## 进一步改进方向

1. **学习因果图结构**：使用图神经网络学习agent之间的因果关系图
2. **因果发现**：从数据中自动发现因果关系
3. **分层因果模型**：区分短期和长期因果效应
4. **因果强化学习**：将因果方法扩展到整个RL流程

