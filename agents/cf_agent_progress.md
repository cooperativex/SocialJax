# CF Debug & Implementation Agent Progress

This file tracks the progress of the CF Agent for implementing and debugging the Counterfactual Regret algorithm.

## Current Status

**Active Task**: CF-IMPL-004 Complete (M4 Regret Calculation implemented)
**Last Session**: 2026-02-21
**Completed Tasks**: 4 / 21
**Pending Tasks**: 17

## Module Dependencies

```
M1 (Generative Model) ✓
├── M2 (Counterfactual Reward) ✓
│   └── M3 (Collective CF Reward) ✓
│       └── M4 (Regret Calculation) ✓
│           └── M5 (Intrinsic Reward)
│               └── M6 (Shaped Reward)
│                   └── M7 (Policy Learning)
│                       └── M9 (Trainer)
│                           └── M10 (Adapters)
│
└── M8 (Causal Attention) [Optional]
```

## Task Summary

### Implementation Tasks (10 tasks)

| ID | Name | Module | Equation | Priority | Status |
|----|------|--------|----------|----------|--------|
| CF-IMPL-001 | 生成模型 Φ_m (RewardModel) | M1 | Eq.6 | high | **DONE** |
| CF-IMPL-002 | 反事实奖励生成 | M2 | Eq.7 | high | **DONE** |
| CF-IMPL-003 | 集体反事实奖励计算 | M3 | Eq.8 | high | **DONE** |
| CF-IMPL-004 | 反事实后悔计算 | M4 | Eq.9 | high | **DONE** |
| CF-IMPL-005 | 内在奖励构造 | M5 | Eq.10 | high | pending |
| CF-IMPL-006 | 奖励塑形 | M6 | Eq.11 | high | pending |
| CF-IMPL-007 | 策略学习 (PPO) | M7 | Eq.12 | high | pending |
| CF-IMPL-008 | 因果注意力机制 | M8 | Appendix | medium | pending |
| CF-IMPL-009 | 完整CF训练循环 | Full | Algo.1 | high | pending |
| CF-IMPL-010 | CF环境适配器 | Env | - | medium | pending |

### Debugging Tasks (5 tasks)

| ID | Name | Priority | Status |
|----|------|----------|--------|
| CF-DEBUG-001 | RewardModel输出形状验证 | high | pending |
| CF-DEBUG-002 | 反事实奖励生成验证 | high | pending |
| CF-DEBUG-003 | 后悔值非负性验证 | high | pending |
| CF-DEBUG-004 | 生成模型训练损失 | high | pending |
| CF-DEBUG-005 | 因果注意力权重 | medium | pending |

### Testing Tasks (3 tasks)

| ID | Name | Priority | Status |
|----|------|----------|--------|
| CF-TEST-001 | Coin Game 1000步冒烟测试 | high | pending |
| CF-TEST-002 | Coin Game 100K步稳定性测试 | medium | pending |
| CF-TEST-003 | Cleanup环境测试 | medium | pending |

### Benchmark Tasks (2 tasks)

| ID | Name | Priority | Status |
|----|------|----------|--------|
| CF-BENCH-001 | CF vs IPPO on Coin Game | medium | pending |
| CF-BENCH-002 | Alpha参数消融实验 | medium | pending |

---

## Sessions

### Session 2026-02-21-1400
**Duration**: ~30 minutes
**Task**: CF-IMPL-004 (反事实后悔计算)
**Status**: completed

### What was done:
- Created `socialjax/algorithms/cf/regret.py` module
- Implemented M4 (Counterfactual Regret) - Eq.9:
  - `compute_counterfactual_regret`: Main regret calculation function
  - `compute_regret_with_best_action`: Returns regret and identifies best prosocial action
  - `compute_normalized_regret`: Normalized regret in [0, 1]
  - `get_regret_statistics`: Statistics for logging (mean, max, zero ratio)
- Updated `__init__.py` with new exports
- Created comprehensive test file: `tests/test_cf/test_regret.py`
  - 25 tests covering:
    - TestComputeCounterfactualRegret (10 tests - shapes, non-negative, optimal, suboptimal, batch sizes)
    - TestComputeRegretWithBestAction (3 tests)
    - TestComputeNormalizedRegret (3 tests)
    - TestGetRegretStatistics (3 tests)
    - TestIntegrationWithCounterfactual (2 tests - pipeline M2->M3->M4)
    - TestEdgeCases (4 tests - single agent, single action, large/small values)
  - All tests passing

### Test criteria verified:
- [x] 后悔值 >= -1e-6 (允许浮点误差)
- [x] 当前动作最优时后悔 ≈ 0
- [x] 存在更优动作时后悔 > 0
- [x] 输出形状: [batch, num_agents]

### Next steps:
- CF-IMPL-005 (内在奖励构造) - Create intrinsic_reward.py module
- Need to implement `compute_intrinsic_reward` function (intrinsic = -regret)

---

### Session 2026-02-21-1200
**Duration**: ~45 minutes
**Task**: CF-IMPL-002, CF-IMPL-003 (反事实奖励生成 & 集体反事实奖励计算)
**Status**: completed

### What was done:
- Created `socialjax/algorithms/cf/counterfactual.py` module
- Implemented M2 (Counterfactual Reward) - Eq.7:
  - `enumerate_counterfactual_actions`: Enumerate all possible actions for an agent
  - `generate_counterfactual_rewards_single_agent`: Generate CF rewards for one agent
  - `generate_counterfactual_rewards_vmap`: Efficient vmap version for all agents
  - `generate_counterfactual_rewards`: Main entry point
- Implemented M3 (Collective CF Reward) - Eq.8:
  - `compute_collective_cf_reward`: Sum rewards for other agents (exclude ego)
  - `compute_actual_collective_reward`: Compute actual collective rewards
  - `get_counterfactual_analysis`: Convenience function combining all operations
- Updated `__init__.py` with new exports
- Fixed bug: reshape was losing `num_agents` dimension in counterfactual reward generation
- Created comprehensive test file: `tests/test_cf/test_counterfactual.py`
  - 26 tests covering:
    - EnumerateCounterfactualActions (7 tests)
    - EnumerateAllAgentsCounterfactualActions (2 tests)
    - GenerateCounterfactualRewards (7 tests - shapes, batch sizes, NaN checks)
    - CollectiveCounterfactualReward (4 tests)
    - ActualCollectiveReward (2 tests)
    - GetCounterfactualAnalysis (2 tests)
    - JITCompilation (2 tests)
    - VmapEfficiency (1 test - verifies vmap matches sequential)
  - All tests passing

### Test criteria verified:
- [x] 枚举所有 |A| 个动作
- [x] 输出形状: [batch, num_actions, num_agents] and [num_agents, action_dim, batch, num_agents]
- [x] 其他agent的动作保持不变
- [x] 使用vmap实现高效计算
- [x] 无NaN/Inf

### Key bug fix:
- Fixed reshape in `generate_counterfactual_rewards_single_agent`:
  - Bug: `obs_flat = obs_expanded.reshape(-1, *obs.shape[2:])` lost `num_agents` dim
  - Fix: `obs_flat = obs_expanded.reshape(-1, *obs.shape[1:])` keeps `num_agents` dim

### Next steps:
- CF-IMPL-004 (反事实后悔计算) - Create regret.py module
- Need to implement `compute_counterfactual_regret` function

---

### Session 2026-02-21-1000
**Duration**: ~30 minutes
**Task**: CF-IMPL-001 (实现生成模型 Φ_m)
**Status**: completed

### What was done:
- Verified JAX environment (melting-jax conda env with 3 CUDA devices)
- Reviewed existing generative_model.py implementation
- Fixed bugs in generative_model.py:
  - Added missing `Any` import in type hints
  - Converted `GenerativeModelLoss` from Flax Module to pure function
- Fixed bugs in __init__.py:
  - Removed imports for non-existent modules
  - Updated exports to use `generative_model_loss` function
- Created comprehensive test file: `tests/test_cf/test_generative_model.py`
  - 33 tests covering:
    - CNNFeatureExtractor (3 tests)
    - RewardModel (14 tests - shapes, batch sizes, agent counts, NaN checks)
    - generative_model_loss function (6 tests)
    - compute_generative_model_loss (3 tests)
    - create_reward_model_train_state (3 tests)
    - Integration with real environment (1 test)
    - Gradient flow verification (2 tests)
  - All tests passing

### Test criteria verified:
- [x] 输出形状正确: [batch, num_agents]
- [x] MSE损失可计算且可微分
- [x] 支持不同batch_size (1, 8, 32, 64)
- [x] 支持不同num_agents (2, 3, 4, 5, 7)
- [x] 无NaN/Inf

---

## Implementation Notes

### Key Design Decisions

1. **Loss function as pure function**: `generative_model_loss` is a pure function instead of Flax Module since it has no learnable parameters. This makes it JAX-friendly for JIT compilation and gradient computation.

2. **CNN architecture**: Uses standard RL visual encoder pattern with 3 conv layers followed by dense projection.

3. **Action encoding**: Uses one-hot encoding for discrete actions, then concatenates with observation features.

### API Reference

#### M1: Generative Model
```python
class RewardModel(nn.Module):
    num_agents: int
    action_dim: int
    cnn_features: Sequence[int] = (32, 32, 32)
    cnn_kernels: Sequence[Tuple[int, int]] = ((5, 5), (3, 3), (3, 3))
    hidden_dim: int = 64
    activation: str = "relu"

    def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            obs: [batch, num_agents, H, W, C]
            actions: [batch, num_agents]
        Returns:
            rewards: [batch, num_agents]
        """

def generative_model_loss(
    predicted_rewards: jnp.ndarray,  # [batch, num_agents]
    actual_rewards: jnp.ndarray,     # [batch, num_agents]
    mask: Optional[jnp.ndarray] = None,
    reduction: str = "mean",
) -> jnp.ndarray:
    """MSE loss for reward prediction (Eq.6)"""

def compute_generative_model_loss(
    params: dict,
    reward_model: RewardModel,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Functional loss computation with model forward pass"""
```

#### M2: Counterfactual Reward Generation
```python
def enumerate_counterfactual_actions(
    actual_actions: jnp.ndarray,  # [batch, num_agents]
    agent_id: int,
    action_dim: int,
) -> jnp.ndarray:
    """
    Enumerate all possible actions for a specific agent.
    Returns: [action_dim, batch, num_agents]
    """

def generate_counterfactual_rewards_single_agent(
    reward_model_apply,
    params: dict,
    obs: jnp.ndarray,          # [batch, num_agents, H, W, C]
    agent_id: int,
    action_dim: int,
    actual_actions: jnp.ndarray,  # [batch, num_agents]
) -> jnp.ndarray:
    """
    Generate CF rewards for one agent. (Eq.7)
    Returns: [action_dim, batch, num_agents]
    """

def generate_counterfactual_rewards_vmap(
    reward_model_apply,
    params: dict,
    action_dim: int,
    obs: jnp.ndarray,          # [batch, num_agents, H, W, C]
    actual_actions: jnp.ndarray,  # [batch, num_agents]
) -> jnp.ndarray:
    """
    Generate CF rewards for all agents using vmap. (Eq.7)
    Returns: [num_agents, action_dim, batch, num_agents]
    """
```

#### M3: Collective Counterfactual Reward
```python
def compute_collective_cf_reward(
    cf_rewards: jnp.ndarray,  # [num_agents, action_dim, batch, num_agents]
    exclude_self: bool = True,
) -> jnp.ndarray:
    """
    Compute collective CF rewards (sum of other agents' rewards). (Eq.8)
    Returns: [num_agents, action_dim, batch]
    """

def compute_actual_collective_reward(
    rewards: jnp.ndarray,  # [batch, num_agents]
) -> jnp.ndarray:
    """
    Compute actual collective reward for each agent.
    Returns: [batch, num_agents]
    """

def get_counterfactual_analysis(
    reward_model_apply,
    params: dict,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    action_dim: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Complete counterfactual analysis combining M2 and M3.
    Returns: (cf_rewards, collective_cf_rewards, actual_collective)
    """
```

#### M4: Counterfactual Regret
```python
def compute_counterfactual_regret(
    collective_cf_rewards: jnp.ndarray,  # [num_agents, action_dim, batch]
    actual_collective: jnp.ndarray,      # [batch, num_agents]
    epsilon: float = 1e-6,
) -> jnp.ndarray:
    """
    Compute counterfactual regret for each agent. (Eq.9)
    Regret_t^i = max_{a^{cf}}[R^{-i,cf}] - R^{-i}
    Returns: [batch, num_agents], always >= 0
    """

def compute_regret_with_best_action(
    collective_cf_rewards: jnp.ndarray,  # [num_agents, action_dim, batch]
    actual_collective: jnp.ndarray,      # [batch, num_agents]
    actual_actions: jnp.ndarray,         # [batch, num_agents]
    epsilon: float = 1e-6,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute regret and identify the best prosocial action.
    Returns: (regret [batch, num_agents], best_actions [batch, num_agents])
    """

def get_regret_statistics(
    regret: jnp.ndarray,  # [batch, num_agents]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute statistics over regret values.
    Returns: (mean_regret, max_regret, zero_regret_ratio) all [num_agents]
    """
```

---

## Known Issues

*None currently*

## Reference

Based on the paper "Resolving Complex Social Dilemmas by Aligning Preferences with Counterfactual Regret" (ICML 2026):

| Equation | Description |
|----------|-------------|
| Eq.6 | Generative Model Loss: L_m = MSE(Φ_m(o,a), r) |
| Eq.7 | Counterfactual Reward: r^{cf} = Φ_m(o, a^{cf}, a^{-i}) |
| Eq.8 | Collective CF Reward: R^{-i,cf} = Σ_{j≠i} r_j^{cf} |
| Eq.9 | Counterfactual Regret: Regret = max(R^{cf}) - R |
| Eq.10 | Intrinsic Reward: r^{in} = -Regret |
| Eq.11 | Shaped Reward: r̂ = r^{ex} + α * r^{in} |
| Eq.12 | Policy Loss: PPO with shaped reward |
