# Level-Based Foraging (LBF) Environment

A JAX-accelerated implementation of the Level-Based Foraging environment, inspired by [semitable/lb-foraging](https://github.com/semitable/lb-foraging).

## Overview

Level-Based Foraging is a multi-agent cooperative-competitive environment where agents must work together to collect food items in a grid world. The key mechanic is that food can only be collected when the sum of adjacent agent levels meets or exceeds the food's level, creating natural cooperation opportunities.

## Key Features

- **JAX-accelerated**: Fully JIT-compiled for high performance
- **Level-based cooperation**: Agents must coordinate to collect high-level food
- **Flexible rewards**: Supports both individual and shared reward modes
- **CNN-compatible observations**: Spatial grid observations suitable for convolutional networks
- **Partial observability**: Agents observe a local window around their position

## Environment Mechanics

### Actions
Each agent can take one of 6 discrete actions:
- `NONE (0)`: Stay in place
- `NORTH (1)`: Move up
- `SOUTH (2)`: Move down
- `WEST (3)`: Move left
- `EAST (4)`: Move right
- `LOAD (5)`: Attempt to collect adjacent food

### Food Collection
Food is collected when:
1. One or more agents adjacent to the food execute the `LOAD` action
2. The sum of those agents' levels ≥ food level

For example:
- Food with level 2 requires 2 agents with level 1, or 1 agent with level 2
- Food with level 3 requires 3 agents with level 1, or combinations totaling ≥3

### Rewards
- **Individual rewards**: Each participating agent receives `food_level × agent_level / total_agent_level`
- **Shared rewards**: All agents receive the same total reward from food collection

### Observations
Agents observe a local `(2×sight+1) × (2×sight+1)` grid centered on their position.

**Observation channels** (for CNN mode):
1. Empty cells
2. Walls (grid boundaries)
3-N. Food levels (one channel per food level, 1 to max_food_level)
N+1. Other agents
N+2. Self

## Usage

### Basic Example

```python
import jax
import socialjax

# Create environment
env = socialjax.make(
    'lb_foraging',
    num_agents=3,
    grid_size=10,
    num_food=4,
    max_food_level=3,
    num_inner_steps=500,
    shared_rewards=False,
    sight=2,
    cnn=True,
    jit=True,
)

# Reset environment
key = jax.random.PRNGKey(0)
obs, state = env.reset(key)

# Take a step
actions = {str(i): 0 for i in range(env.num_agents)}  # All agents do nothing
obs, state, rewards, dones, info = env.step(key, state, actions)
```

### Parameters

- `num_agents` (int, default=3): Number of agents
- `grid_size` (int, default=10): Size of the square grid
- `num_food` (int, default=3): Number of food items in the environment
- `max_food_level` (int, default=3): Maximum food level (agent levels are fixed at 1)
- `num_inner_steps` (int, default=500): Episode length
- `shared_rewards` (bool, default=False): Whether to use shared rewards
- `sight` (int, default=3): Observation radius (agents see `2×sight+1` cells)
- `cnn` (bool, default=True): Use CNN-compatible observations
- `jit` (bool, default=True): JIT-compile environment methods

## Comparison with Original LBF

This is a **simplified** implementation focusing on core mechanics:

### Included
✅ Level-based cooperation mechanism
✅ Grid world navigation
✅ Food collection with adjacency requirements
✅ Individual and shared reward modes
✅ Partial observability
✅ Collision avoidance (agents can't occupy same cell)

### Simplified
- Agent levels are fixed at 1 (food levels vary from 1 to max_food_level)
- No food respawning (food disappears when collected)
- Simpler observation space (grid-based only, no feature vectors)
- Fixed grid boundaries (walls on borders)
- No sight-based food spawning constraints

### Differences from Social Dilemma Environments

Unlike Harvest/Cleanup environments in SocialJax:
- **Cooperation requirement**: Enforced by level mechanics (not emergent)
- **Resource management**: No regeneration dynamics
- **Social dilemma**: Less emphasis on tragedy of the commons
- **Episode structure**: Fixed food count, finite resources

## Research Applications

This environment is suitable for studying:
- **Explicit cooperation**: Level requirements force multi-agent coordination
- **Credit assignment**: How to fairly distribute rewards among cooperating agents
- **Individual vs. collective incentives**: Compare `shared_rewards=True/False`
- **Multi-agent exploration**: Finding and coordinating at food locations
- **MARL algorithm benchmarking**: Standard testbed like the original LBF

## Example Algorithms

The environment works with all SocialJax algorithms:
- **IPPO**: Independent learning baseline
- **MAPPO**: Centralized value function
- **IQL**: Independent Q-learning
- **SVO**: Social value orientation

## Citation

Original LBF environment:
```bibtex
@inproceedings{christianos2020shared,
  title={Shared Experience Actor-Critic for Multi-Agent Reinforcement Learning},
  author={Christianos, Filippos and Schäfer, Lukas and Albrecht, Stefano V},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```

## License

This implementation follows the SocialJax license structure.
