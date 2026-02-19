# 快速启用因果Reward Model

## 方法1：完全启用（推荐）

在 `cf_cnn_coin.yaml` 中修改：

```yaml
"USE_CAUSAL_REWARD_MODEL": True  # 改为 True
```

其他参数已设置为推荐值，可以直接使用。

## 方法2：逐步启用

### 步骤1：只启用因果注意力（轻量级）

```yaml
"USE_CAUSAL_REWARD_MODEL": True
"USE_CAUSAL_ATTENTION": True
"USE_CAUSAL_TRAINING": False  # 先禁用反事实训练
```

### 步骤2：如果效果好，再启用反事实训练

```yaml
"USE_CAUSAL_REWARD_MODEL": True
"USE_CAUSAL_ATTENTION": True
"USE_CAUSAL_TRAINING": True  # 启用反事实训练
"CF_NUM_SAMPLES": 5  # 可以调整（3-10之间）
```

## 配置参数说明

### 核心参数

- `USE_CAUSAL_REWARD_MODEL`: 主开关，设为 `True` 启用因果模型
- `USE_CAUSAL_ATTENTION`: 是否使用因果注意力（推荐开启）
- `USE_CAUSAL_TRAINING`: 是否使用反事实训练（会增加训练时间）

### 调优参数

- `NUM_ATTENTION_HEADS`: 注意力头数（2-8，默认4）
- `CF_NUM_SAMPLES`: 反事实样本数（3-10，默认5，越大越慢）
- `CAUSAL_CONSISTENCY_WEIGHT`: 因果损失权重（0.05-0.2，默认0.1）

### 高级参数

- `CAUSAL_MASK`: 如果知道agent之间的因果关系，可以定义掩码
  ```yaml
  "CAUSAL_MASK": [[0,0,0], [1,0,0], [1,1,0]]  # 3个agent的例子
  ```

## 性能优化

如果训练速度变慢：

1. **减少反事实样本**：
   ```yaml
   "CF_NUM_SAMPLES": 3  # 从5改为3
   ```

2. **减少注意力头数**：
   ```yaml
   "NUM_ATTENTION_HEADS": 2  # 从4改为2
   ```

3. **禁用反事实训练**：
   ```yaml
   "USE_CAUSAL_TRAINING": False
   ```

## 监控指标

启用后，在wandb中关注：

- `causal_loss`: 应该逐渐降低
- `reward_prediction_accuracy`: 应该提高
- `reward_correlation`: 应该接近1.0
- `use_causal_model`: 应该为1.0（确认已启用）

## 示例配置

### 最小配置（最快）
```yaml
"USE_CAUSAL_REWARD_MODEL": True
"USE_CAUSAL_ATTENTION": True
"USE_CAUSAL_TRAINING": False
"NUM_ATTENTION_HEADS": 2
```

### 平衡配置（推荐）
```yaml
"USE_CAUSAL_REWARD_MODEL": True
"USE_CAUSAL_ATTENTION": True
"USE_CAUSAL_TRAINING": True
"NUM_ATTENTION_HEADS": 4
"CF_NUM_SAMPLES": 5
```

### 完整配置（最佳效果，较慢）
```yaml
"USE_CAUSAL_REWARD_MODEL": True
"USE_CAUSAL_ATTENTION": True
"USE_CAUSAL_TRAINING": True
"NUM_ATTENTION_HEADS": 4
"CF_NUM_SAMPLES": 10
"CAUSAL_CONSISTENCY_WEIGHT": 0.15
```

