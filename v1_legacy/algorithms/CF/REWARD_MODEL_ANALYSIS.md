# Reward Model 训练与 Reward Shaping 分析

## 1. Reward Model 架构

Reward Model 接收所有 agent 的观察和动作，预测每个 agent 的奖励值：

```69:100:algorithms/CF/cf_cnn_coin.py
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
        
        # 输出每个agent的奖励值 [batch, num_agents]
        x = nn.Dense(num_agents, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = x.reshape(batch_size, num_agents)
        return x
```

**关键点**：
- 输入：所有 agent 的观察 + 所有 agent 的动作
- 输出：每个 agent 的预测奖励 `[batch, num_agents]`
- 架构：CNN 提取观察特征 → 拼接动作 one-hot → Dense 层输出奖励

## 2. Reward Model 训练流程

### 2.1 训练目标
使用**真实环境奖励**作为监督信号，学习预测每个 agent 的个体奖励：

```481:522:algorithms/CF/cf_cnn_coin.py
def reward_loss_fn(reward_params, traj_batch):
    action_for_reward = jnp.reshape(traj_batch.action, (-1, env.num_agents))
    obs_for_reward = jnp.reshape(traj_batch.obs, (-1, env.num_agents, *(traj_batch.obs.shape[1:])))
    
    # 获取预测奖励值 [batch, num_agents]
    pred_rewards = reward_model.apply(reward_params, obs_for_reward, action_for_reward)
    
    # 真实奖励
    true_reward = jnp.reshape(traj_batch.reward, (-1, env.num_agents))
    
    # 根据训练阶段调整学习权重
    reward_train_end_step = config.get("REWARD_TRAIN_END_STEP", 1200)
    early_train_boost = config.get("EARLY_TRAIN_BOOST_STEPS", 400)
    
    def get_loss_weight():
        return jax.lax.cond(
            update_step < early_train_boost,
            lambda: 3.0,  # 前400步：加强学习权重
            lambda: jax.lax.cond(
                update_step < reward_train_end_step,
                lambda: 1.5,  # 400-1200步：中等权重
                lambda: 1.0   # 1200步后：正常权重
            )
        )
    
    loss_weight = get_loss_weight()

    # 使用MSE损失
    mse_loss = jnp.mean((pred_rewards - true_reward) ** 2)
    
    # 计算MAE（用于监控）
    mae = jnp.mean(jnp.abs(pred_rewards - true_reward))
    
    loss = loss_weight * mse_loss
    return loss, {
        "reward_loss": loss, 
        "reward_mae": mae,
        "pred_reward_mean": jnp.mean(pred_rewards),
        "true_reward_mean": jnp.mean(true_reward),
        "loss_weight": loss_weight,
        "mse_loss": mse_loss
    }
```

**训练阶段**：
- **0-200 步**：loss_weight = 3.0（快速学习）
- **200-1200 步**：loss_weight = 1.5（正常学习）
- **1200 步后**：loss_weight = 1.0（但可能已冻结）

### 2.2 训练停止条件

```587:596:algorithms/CF/cf_cnn_coin.py
# 先计算一次损失来检查收敛状态
_, temp_loss_info = reward_loss_fn(reward_state.params, traj_batch)
reward_mae_threshold = config.get("REWARD_MAE_THRESHOLD", 0.1)
reward_converged = temp_loss_info["reward_mae"] < reward_mae_threshold

reward_state, reward_total_loss, reward_loss_info = jax.lax.cond(
    jnp.logical_and(update_step < reward_train_end_step, jnp.logical_not(reward_converged)),
    train_reward_model,  # 在1200步内且未收敛：继续训练reward model
    freeze_reward_model   # 1200步后或已收敛：冻结reward model
)
```

**停止条件**：
- `update_step >= 1200` **或** `MAE < 0.1`
- 满足任一条件即冻结 reward model

## 3. Reward Shaping 使用方式

### 3.1 计算 Collective Reward

```326:330:algorithms/CF/cf_cnn_coin.py
# test collective reward  
pred_rewards = reward_state.apply_fn(reward_state.params, obs_for_cf, actions_for_cf)
# pred_rewards shape: [batch, num_agents]，直接是奖励值
# 对所有agent求和得到集体奖励
pred_collective_rewards = jnp.sum(pred_rewards, axis=1)  # [batch]
```

**关键**：将每个 agent 的预测奖励**求和**得到集体奖励。

### 3.2 渐进式奖励混合

```332:349:algorithms/CF/cf_cnn_coin.py
shaped_rewards = info["shaped_rewards"].squeeze(-1)

# 渐进式奖励混合：从shaped_rewards平滑过渡到pred_collective_rewards
reward_switch_start = config.get("REWARD_SWITCH_START", 500)  # 开始混合的步数
reward_switch_end = config.get("REWARD_SWITCH_END", 1000)    # 完全切换的步数

# 计算混合权重：从0平滑增加到1
mix_weight = jnp.clip(
    (update_step - reward_switch_start) / (reward_switch_end - reward_switch_start),
    0.0, 1.0
)

# 确保collective_rewards有正确的形状
collective_rewards_expanded = jnp.expand_dims(pred_collective_rewards, axis=1)  # [batch, 1]
collective_rewards_broadcast = jnp.broadcast_to(collective_rewards_expanded, shaped_rewards.shape)

# 渐进式混合：(1-α) * shaped_rewards + α * pred_collective_rewards
cf_rewards = (1.0 - mix_weight) * shaped_rewards + mix_weight * collective_rewards_broadcast
```

**混合策略**（根据配置：200-400 步）：
- **0-200 步**：`mix_weight = 0`，完全使用 `shaped_rewards`
- **200-400 步**：`mix_weight` 从 0 线性增加到 1
- **400+ 步**：`mix_weight = 1`，完全使用 `pred_collective_rewards`

### 3.3 在 GAE 中使用

```374:395:algorithms/CF/cf_cnn_coin.py
def _calculate_gae(traj_batch, last_val):
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value,_ = gae_and_next_value
        done, value, reward, org_cf_regret = (
            transition.done,
            transition.value,
            transition.reward,
            transition.cf_regret,
        )

        reward_mean = jnp.mean(reward, axis=0)
        # reward_std = jnp.std(reward, axis=0) + 1e-8
        reward = (reward - reward_mean)# / reward_std
        cf_regret_mean = jnp.mean(org_cf_regret, axis=0)
        cf_regret = (org_cf_regret - cf_regret_mean)# / (cf_regret.std() + 1e-8)
        #cf_shaped_reward = reward - config["REWARD_ALPHA"] * cf_regret
        
        delta = org_cf_regret + config["GAMMA"] * next_value * (1 - done) - value
        gae = (
            delta
            + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
        )
```

**关键发现**：
- `delta` 使用的是 `org_cf_regret`（即 `cf_rewards`），**不是**原始 `reward`
- `cf_rewards` 是混合后的奖励（`shaped_rewards` → `pred_collective_rewards`）

## 4. 潜在问题分析

### 问题 1：Reward Model 学习目标 vs 实际使用不匹配

**训练时**：
- Reward Model 学习预测**个体奖励** `[batch, num_agents]`
- 监督信号：`true_reward`（环境给出的个体奖励）

**使用时**：
- 将所有 agent 的预测奖励**求和**：`pred_collective_rewards = sum(pred_rewards)`
- 这个集体奖励被广播给所有 agent 使用

**后果**：
- Reward Model 没有学习"集体奖励"的概念
- 它学习的是个体奖励，但被当作集体奖励使用
- 如果个体奖励总和 ≠ 集体奖励（在某些环境中），会导致预测偏差

### 问题 2：Reward Model 在早期可能不准确

**时间线**：
- **0-200 步**：Reward Model 刚开始训练，预测不准确
- **200-400 步**：开始混合使用 `pred_collective_rewards`，但模型可能仍不准确
- **400+ 步**：完全切换到 `pred_collective_rewards`，但模型可能仍未收敛

**后果**：
- 在 200-400 步之间，如果 reward model 预测不准确，会导致：
  - `pred_collective_rewards` 与真实集体奖励差异大
  - 策略学习到错误的奖励信号
  - Reward curve 可能出现**突然下降或波动**

### 问题 3：Reward Model 冻结后不再更新

**配置**：
- `REWARD_TRAIN_END_STEP = 1200`
- `REWARD_MAE_THRESHOLD = 0.1`

**问题**：
- 如果 reward model 在 1200 步前就达到 MAE < 0.1，会被冻结
- 但策略在继续学习，环境分布可能发生变化（distribution shift）
- 冻结的 reward model 无法适应新的策略分布
- 可能导致 reward curve 在后期**停滞或下降**

### 问题 4：Collective Reward 广播给所有 Agent

```345:346:algorithms/CF/cf_cnn_coin.py
collective_rewards_expanded = jnp.expand_dims(pred_collective_rewards, axis=1)  # [batch, 1]
collective_rewards_broadcast = jnp.broadcast_to(collective_rewards_expanded, shaped_rewards.shape)
```

**问题**：
- 所有 agent 收到**相同的**集体奖励
- 这消除了个体差异，可能导致：
  - 所有 agent 学习相同的策略
  - 缺乏多样性
  - 在需要分工的环境中表现不佳

### 问题 5：Counterfactual Regret 计算了但未使用

```299:324:algorithms/CF/cf_cnn_coin.py
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
```

**问题**：
- Counterfactual regret 被计算并存储在 `transition.cf_regret` 中
- 但在 GAE 计算中，使用的是 `org_cf_regret`（实际上是 `cf_rewards`，不是 `cf_regrets`）
- Counterfactual regret **没有被使用**，这是计算浪费

## 5. Reward Curve 可能的表现

基于以上分析，reward curve 可能出现以下模式：

### 模式 1：早期下降（200-400 步）
- **原因**：Reward Model 不准确时开始使用 `pred_collective_rewards`
- **表现**：Reward 在 200-400 步之间突然下降

### 模式 2：中期波动（400-1200 步）
- **原因**：Reward Model 在训练中，预测不稳定
- **表现**：Reward 曲线波动较大

### 模式 3：后期停滞（1200+ 步）
- **原因**：Reward Model 冻结，无法适应策略变化
- **表现**：Reward 不再提升或下降

### 模式 4：整体偏低
- **原因**：Collective reward 广播导致所有 agent 学习相同策略，缺乏多样性
- **表现**：Reward 始终低于预期

## 6. 建议的修复方案

### 方案 1：延迟 Reward Shaping 切换
- 将 `REWARD_SWITCH_START` 推迟到 reward model 收敛后
- 例如：`REWARD_SWITCH_START = REWARD_TRAIN_END_STEP`

### 方案 2：持续训练 Reward Model
- 不要冻结 reward model，让它持续适应策略变化
- 降低学习率，但保持更新

### 方案 3：使用个体奖励而非集体奖励
- 不要求和，直接使用每个 agent 的预测奖励
- 保持个体差异

### 方案 4：使用 Counterfactual Regret
- 如果计算了 counterfactual regret，应该使用它
- 修改 GAE 计算，使用 `cf_regrets` 而不是 `cf_rewards`

### 方案 5：Reward Model 学习集体奖励
- 修改训练目标，让 reward model 直接学习集体奖励
- 而不是学习个体奖励后求和

