# Reward Model 预测准确率改进方案

## 问题分析

从训练曲线观察到：
1. **reward_prediction_accuracy** 在400-450步达到峰值0.76，然后下降
2. **returned_episode_original_returns** 在400-420步达到峰值后逐渐下降
3. 680步后进入高波动期

### 根本原因

1. **Reward Model过早停止训练**
   - 当 `mix_weight > 0.8` 或保护期结束后停止训练
   - 但此时策略还在变化，数据分布也在变化
   - Reward Model无法适应新的分布，导致预测准确率下降

2. **数据分布偏移（Distribution Shift）**
   - 策略改变 → 观察和动作分布改变
   - Reward Model已停止训练 → 无法适应新分布
   - 形成负反馈循环

3. **损失函数不够鲁棒**
   - 使用MSE损失，对异常值敏感
   - 集体奖励损失权重（0.1）太小，无法有效约束集体奖励预测

4. **收敛判断过于严格**
   - MAE阈值0.1可能过于严格
   - 只考虑绝对值，不考虑趋势和相关性

## 改进方案

### 改进1：使用更鲁棒的损失函数（Huber损失）

**问题**：MSE损失对异常值敏感，当有少量异常样本时，会导致模型过度拟合这些异常值。

**解决方案**：
```python
def huber_loss(pred, true):
    error = pred - true
    abs_error = jnp.abs(error)
    quadratic = jnp.clip(abs_error, 0, huber_delta)
    linear = abs_error - quadratic
    return jnp.mean(quadratic ** 2 + huber_delta * linear)
```

**优势**：
- 对小误差使用二次损失（MSE），对大误差使用线性损失
- 对异常值更鲁棒，不会过度拟合异常样本
- 可以通过 `HUBER_DELTA` 参数控制鲁棒性

### 改进2：增加集体奖励损失权重

**问题**：实际使用时，reward model预测的个体奖励会被求和得到集体奖励。如果集体奖励预测不准确，会直接影响策略学习。

**解决方案**：
- 将 `COLLECTIVE_LOSS_WEIGHT` 从 0.1 增加到 0.5
- 确保集体奖励预测的准确性

**优势**：
- 直接优化实际使用的目标（集体奖励）
- 减少求和带来的误差累积

### 改进3：添加相关性损失

**问题**：即使绝对误差较大，只要预测和真实奖励的趋势一致，也能帮助策略学习。但MSE损失只关注绝对误差。

**解决方案**：
```python
correlation = jnp.sum(pred_centered * true_centered) / (jnp.sqrt(jnp.sum(pred_centered ** 2) * jnp.sum(true_centered ** 2)) + 1e-8)
correlation_loss = (1.0 - correlation) ** 2  # 鼓励高相关性
```

**优势**：
- 确保预测和真实奖励的趋势一致
- 即使绝对误差较大，只要趋势正确，也能帮助策略学习
- 通过 `CORRELATION_LOSS_WEIGHT` 控制权重（默认0.1）

### 改进4：更智能的训练策略

**问题**：当策略改变时，数据分布也会改变，reward model需要持续适应。

**解决方案**：
```python
# 检查MAE是否仍然较高
mae_still_high = temp_loss_info["reward_mae"] > config.get("REWARD_MAE_CONTINUE_THRESHOLD", 0.2)

# 检查相关性是否较低
correlation_low = temp_loss_info.get("reward_correlation", 0.0) < config.get("REWARD_CORRELATION_THRESHOLD", 0.7)

# 允许持续训练
allow_continuous_training = config.get("ALLOW_CONTINUOUS_REWARD_TRAINING", True)
```

**优势**：
- 不仅看MAE绝对值，还看相关性
- 允许reward model持续训练以适应分布偏移
- 更宽松的收敛判断（MAE > 0.2 或 correlation < 0.7）

### 改进5：自适应学习率调度

**问题**：当开始使用reward model后，如果学习率太大，会导致训练不稳定。

**解决方案**：
```python
def reward_lr_schedule(count):
    if update_step_approx < reward_switch_start:
        return reward_lr  # 正常学习率
    elif update_step_approx < reward_switch_end:
        return reward_lr * (1.0 - 0.5 * mix_weight)  # 线性降低到50%
    else:
        return reward_lr * 0.1  # 降低到10%，持续适应分布偏移
```

**优势**：
- 在过渡期降低学习率，稳定训练
- 完全使用后使用更小的学习率（10%），持续适应分布偏移
- 避免负反馈循环，同时保持适应性

## 配置参数

新增/修改的配置参数：

```yaml
# 损失函数相关
USE_HUBER_LOSS: true  # 是否使用Huber损失（默认true）
HUBER_DELTA: 1.0  # Huber损失的阈值
COLLECTIVE_LOSS_WEIGHT: 0.5  # 集体奖励损失权重（从0.1增加到0.5）
CORRELATION_LOSS_WEIGHT: 0.1  # 相关性损失权重

# 训练策略相关
ALLOW_CONTINUOUS_REWARD_TRAINING: true  # 允许持续训练（默认true）
REWARD_MAE_CONTINUE_THRESHOLD: 0.2  # MAE继续训练阈值（从0.1放宽到0.2）
REWARD_CORRELATION_THRESHOLD: 0.7  # 相关性阈值（低于此值继续训练）
USE_ADAPTIVE_REWARD_LR: true  # 使用自适应学习率（默认true）
```

## 预期效果

1. **提高预测准确率**：
   - Huber损失减少异常值影响
   - 相关性损失确保趋势一致
   - 集体奖励损失权重增加，直接优化使用目标

2. **提高稳定性**：
   - 自适应学习率在过渡期稳定训练
   - 更宽松的收敛判断，避免过早停止

3. **提高适应性**：
   - 允许持续训练，适应分布偏移
   - 更小的学习率（10%）在完全使用后持续微调

## 监控指标

新增的监控指标：
- `loss_individual`: 个体奖励损失
- `loss_collective`: 集体奖励损失
- `correlation_loss`: 相关性损失
- `use_huber_loss`: 是否使用Huber损失
- `reward_model_is_training`: Reward Model是否在训练

## 使用建议

1. **初始训练**：使用默认配置，观察效果
2. **如果预测准确率仍然不高**：
   - 增加 `COLLECTIVE_LOSS_WEIGHT` 到 0.8-1.0
   - 增加 `CORRELATION_LOSS_WEIGHT` 到 0.2-0.3
   - 调整 `HUBER_DELTA`（越小越鲁棒，但可能欠拟合）
3. **如果训练不稳定**：
   - 降低 `REWARD_LR`
   - 确保 `USE_ADAPTIVE_REWARD_LR` 为 true
4. **如果准确率下降**：
   - 确保 `ALLOW_CONTINUOUS_REWARD_TRAINING` 为 true
   - 放宽 `REWARD_MAE_CONTINUE_THRESHOLD` 到 0.3-0.5
   - 降低 `REWARD_CORRELATION_THRESHOLD` 到 0.6

