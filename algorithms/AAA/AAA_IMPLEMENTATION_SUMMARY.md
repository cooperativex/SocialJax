# AAA (Advantage Alignment Actor) Implementation Summary

## Algorithm Overview

AAA (Advantage Alignment Actor) is a novel opponent shaping method that modifies PPO's advantage function to foster mutually beneficial strategies in multi-agent reinforcement learning.

**Key Innovation**: Instead of purely optimizing individual rewards, AAA aligns each agent's advantages with those of its opponents, steering towards cooperative equilibria.

## Mathematical Formulation

### AAA Advantage Modification Formula

```
A*(s_t, a_t, b_t) = A¹(s_t, a_t, b_t) + β·γ·(Σ_{k<t} γ^{t-k} A¹(s_k, a_k, b_k))·A²(s_t, a_t, b_t)
```

**Components**:
- **A¹**: The agent's standard advantage estimate
- **A²**: The opponent's advantage (sum of all other agents' advantages)
- **β**: Scaling parameter controlling opponent advantage influence (default: 0.1)
- **γ**: Discount factor (from config: 0.99)
- **Σ_{k<t} γ^{t-k} A¹(s_k, a_k, b_k)**: Cumulative discounted past advantages

## Implementation Details

### 1. Modified Advantage Calculation

Located in `_calculate_gae()` function ([aaa_cnn_harvest_common.py:304-384](aaa_cnn_harvest_common.py#L304-L384)):

**Step 1**: Compute standard GAE advantages
```python
# Standard GAE calculation
_, advantages = jax.lax.scan(_get_advantages, ...)
```

**Step 2**: Reshape for agent separation
```python
# Shape: (num_steps, num_actors) -> (num_steps, num_agents, num_envs)
advantages_reshaped = advantages.reshape(num_steps, env.num_agents, config["NUM_ENVS"])
```

**Step 3**: For each agent, apply AAA formula
```python
def apply_aaa_to_agent(agent_idx, advantages_reshaped):
    # Get agent's advantages
    agent_adv = advantages_reshaped[:, agent_idx, :]

    # Compute cumulative past advantages: Σ_{k<t} γ^{t-k} A¹(s_k, a_k, b_k)
    _, cumulative_past_adv = jax.lax.scan(cumsum_discounted, ...)

    # Get opponent advantages (sum of all other agents)
    opponent_mask = jnp.ones(env.num_agents).at[agent_idx].set(0)
    opponent_adv = jnp.sum(advantages_reshaped * opponent_mask[None, :, None], axis=1)

    # AAA formula: A* = A¹ + β·γ·(cumulative_past_A¹)·A²
    modified_adv = agent_adv + config["AAA_BETA"] * config["GAMMA"] * cumulative_past_adv * opponent_adv

    return modified_adv
```

**Step 4**: Apply to all agents and reshape back
```python
for i in range(env.num_agents):
    modified_adv = apply_aaa_to_agent(i, advantages_reshaped)
    modified_advantages_list.append(modified_adv)

modified_advantages = jnp.stack(modified_advantages_list, axis=1)
modified_advantages = modified_advantages.reshape(num_steps, config["NUM_ACTORS"])
```

### 2. Configuration

**File**: `config/aaa_cnn_harvest_common.yaml`

**Key AAA Parameter**:
```yaml
"AAA_BETA": 0.1  # Scaling parameter for opponent advantage influence
```

**Environment Setup**:
```yaml
"ENV_KWARGS":
  "num_agents": 7
  "num_inner_steps": 1000
  "shared_rewards": False  # AAA uses individual rewards
  "cnn": True
  "jit": True
```

**WandB Tracking**:
```yaml
"PROJECT": "socialjax_aaa"
"WANDB_TAGS":
  - AAA
  - ADVANTAGE_ALIGNMENT
  - INDIVIDUAL_REWARD
```

### 3. Network Architecture

AAA uses the same network architecture as MAPPO:
- **Actor**: CNN + MLP policy network (local observations)
- **Critic**: CNN + MLP value network (global state)

No architectural changes needed - the innovation is purely in the advantage modification.

## Differences from MAPPO

| Aspect | MAPPO | AAA |
|--------|-------|-----|
| **Advantage** | Standard GAE | Modified with opponent advantages |
| **Cooperation** | Implicit (shared value) | Explicit (opponent-aware) |
| **Formula** | `A(s,a)` | `A(s,a) + β·γ·(cumulative_A)·A_opponent` |
| **Equilibria** | Nash | Mutually beneficial |
| **Complexity** | O(1) per agent | O(n) per agent (n=num_agents) |

## Key Benefits

1. **Opponent-Aware Learning**: Agents explicitly consider opponent advantages
2. **Mutually Beneficial Outcomes**: Steers towards win-win situations
3. **Simple Integration**: Only modifies advantage calculation, keeps PPO structure
4. **Tunable Cooperation**: β parameter controls opponent influence strength

## Hyperparameter Tuning

**β (AAA_BETA)** - Opponent influence scaling:
- **β = 0**: Reduces to standard PPO (no opponent consideration)
- **β = 0.1** (default): Moderate opponent influence
- **β = 0.5**: Strong opponent influence
- **β > 1**: May destabilize training

**Recommended Values**:
- Competitive environments: 0.05 - 0.2
- Cooperative environments: 0.1 - 0.5
- Mixed environments: 0.1 - 0.3

## Running AAA

```bash
cd algorithms/AAA
conda activate SocialJax
export PYTHONPATH=/c/Phd_study/SocialJax:$PYTHONPATH
python aaa_cnn_harvest_common.py
```

## Expected Behavior

Compared to MAPPO, AAA should:
- ✅ Achieve higher team rewards in cooperative tasks
- ✅ Learn more stable joint policies
- ✅ Converge to mutually beneficial equilibria
- ✅ Show improved coordination between agents

## References

- **Paper**: Advantage Alignment Algorithms
- **GitHub**: https://github.com/jduquevan/advantage-alignment
- **Environment**: SocialJax Harvest Common Open (7 agents, mixed-incentive)

## Implementation Checklist

- [x] Modified advantage calculation with AAA formula
- [x] Added AAA_BETA hyperparameter
- [x] Updated config file for AAA
- [x] Updated WandB tags and project name
- [x] Updated Hydra config name
- [x] Added comprehensive documentation
- [ ] Tested training convergence
- [ ] Compared against MAPPO baseline
- [ ] Tuned β parameter for optimal performance
