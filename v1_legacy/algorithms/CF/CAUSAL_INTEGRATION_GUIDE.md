# 因果Reward Model集成指南

## ✅ 集成完成

因果Reward Model已成功集成到`cf_cnn_coin.py`中。现在可以通过配置参数来启用或禁用因果方法。

## 使用方法

### 1. 基本配置

在配置文件中添加以下参数来启用因果Reward Model：

```yaml
# 启用因果Reward Model
USE_CAUSAL_REWARD_MODEL: true

# 因果注意力相关配置
USE_CAUSAL_ATTENTION: true  # 是否使用因果注意力机制
NUM_ATTENTION_HEADS: 4  # 注意力头数（默认4）

# 因果训练相关配置
USE_CAUSAL_TRAINING: true  # 是否使用反事实数据增强和因果损失
CF_NUM_SAMPLES: 5  # 反事实样本数量（默认5）
CAUSAL_CONSISTENCY_WEIGHT: 0.1  # 因果一致性损失权重（默认0.1）

# 可选的因果掩码（如果知道agent之间的因果关系结构）
CAUSAL_MASK: null  # 或定义如：[[0, 0, 0], [1, 0, 0], [1, 1, 0]]
```

### 2. 配置示例

#### 示例1：完全启用因果方法

```yaml
USE_CAUSAL_REWARD_MODEL: true
USE_CAUSAL_ATTENTION: true
NUM_ATTENTION_HEADS: 4
USE_CAUSAL_TRAINING: true
CF_NUM_SAMPLES: 5
CAUSAL_CONSISTENCY_WEIGHT: 0.1
```

#### 示例2：只使用因果注意力，不使用反事实训练

```yaml
USE_CAUSAL_REWARD_MODEL: true
USE_CAUSAL_ATTENTION: true
NUM_ATTENTION_HEADS: 4
USE_CAUSAL_TRAINING: false  # 禁用反事实训练
```

#### 示例3：使用预定义的因果掩码

```yaml
USE_CAUSAL_REWARD_MODEL: true
USE_CAUSAL_ATTENTION: true
NUM_ATTENTION_HEADS: 4
USE_CAUSAL_TRAINING: true
CAUSAL_MASK: 
  - [0, 0, 0]  # agent 0 不影响其他agent
  - [1, 0, 0]  # agent 1 影响 agent 0
  - [1, 1, 0]  # agent 2 影响 agent 0 和 1
```

### 3. 监控指标

集成后，以下新的监控指标会被记录到wandb：

- `causal_loss`: 因果一致性损失值
- `use_causal_model`: 是否使用因果模型（1.0表示是，0.0表示否）
- `loss_individual`: 个体奖励损失
- `loss_collective`: 集体奖励损失
- `correlation_loss`: 相关性损失

### 4. 代码变更说明

#### 主要变更：

1. **导入因果模型**：
   ```python
   from algorithms.CF.causal_reward_model import (
       CausalRewardModel, 
       generate_counterfactual_samples_jax, 
       causal_consistency_loss
   )
   ```

2. **条件使用因果模型**：
   - 根据`USE_CAUSAL_REWARD_MODEL`配置选择使用`CausalRewardModel`或标准`RewardModel`
   - 自动处理初始化参数（包括可选的`causal_mask`）

3. **统一接口函数**：
   - `apply_reward_model()`函数统一处理reward model调用
   - 自动处理因果模型的`causal_mask`参数

4. **因果损失集成**：
   - 在`reward_loss_fn`中添加了反事实样本生成
   - 计算因果一致性损失并加入总损失

### 5. 向后兼容性

✅ **完全向后兼容**：
- 默认情况下（`USE_CAUSAL_REWARD_MODEL: false`），使用标准的`RewardModel`
- 所有现有配置和代码都能正常工作
- 如果因果模型不可用（导入失败），会自动回退到标准模型

### 6. 性能考虑

- **因果注意力**：增加少量计算开销（注意力机制）
- **反事实训练**：会增加训练时间（生成反事实样本）
  - 可以通过`CF_NUM_SAMPLES`控制样本数量
  - 可以通过`USE_CAUSAL_TRAINING: false`完全禁用

### 7. 故障排除

#### 问题1：导入错误
```
ImportError: cannot import name 'CausalRewardModel'
```

**解决方案**：
- 确保`causal_reward_model.py`文件在正确的位置
- 检查文件路径和导入语句
- 代码会自动回退到标准模型

#### 问题2：形状不匹配
```
ValueError: shapes mismatch
```

**解决方案**：
- 检查`CAUSAL_MASK`的形状是否与`num_agents`匹配
- 确保`NUM_ATTENTION_HEADS`是合理的值（建议2-8）

#### 问题3：训练速度变慢
**解决方案**：
- 减少`CF_NUM_SAMPLES`（例如从5改为3）
- 设置`USE_CAUSAL_TRAINING: false`禁用反事实训练
- 减少`NUM_ATTENTION_HEADS`（例如从4改为2）

### 8. 最佳实践

1. **首次使用**：
   - 先使用默认配置测试
   - 观察`causal_loss`和`reward_mae`的变化
   - 如果效果不好，可以调整`CAUSAL_CONSISTENCY_WEIGHT`

2. **调优顺序**：
   - 先启用`USE_CAUSAL_ATTENTION`（只使用注意力机制）
   - 如果效果好，再启用`USE_CAUSAL_TRAINING`（添加反事实训练）
   - 最后考虑添加`CAUSAL_MASK`（如果有先验知识）

3. **监控重点**：
   - `reward_prediction_accuracy`: 预测准确率
   - `reward_correlation`: 预测和真实奖励的相关性
   - `causal_loss`: 因果一致性损失（应该逐渐降低）

### 9. 预期效果

使用因果方法后，预期会看到：
- ✅ 更高的`reward_prediction_accuracy`
- ✅ 更高的`reward_correlation`
- ✅ 更稳定的训练曲线
- ✅ 更好的泛化能力

## 总结

因果Reward Model已完全集成，可以通过简单的配置参数启用。所有功能都是可选的，不会影响现有代码的正常运行。

