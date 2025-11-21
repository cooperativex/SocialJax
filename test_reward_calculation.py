"""
Test LBF reward calculation logic
"""
import jax
import jax.numpy as jnp
import numpy as np
import socialjax

print("=" * 60)
print("Testing LBF Reward Calculation")
print("=" * 60)

# Create simple environment
env = socialjax.make(
    "lb_foraging",
    num_agents=3,
    num_food=1,  # Just 1 food for easy testing
    max_food_level=3,
    max_player_level=2,
    num_inner_steps=10,
    sight=7,
    jit=False,  # Disable JIT for debugging
    cnn=True,
    normalize_reward=True,
    load_penalty=0.1,
)

# Reset and manually set state for controlled testing
key = jax.random.PRNGKey(42)
obs, state = env.reset(key)

print("\n[Test 1] Manual Setup: 1 food at (3,3) with level 2")
print("Agent positions and levels:")

# Manually create a controlled state
# Food at (3, 3) with level 2
food_grid = jnp.zeros(env.grid_shape, dtype=jnp.int32)
food_grid = food_grid.at[3, 3].set(12)  # level 2 food (10 + 2)

# Agent 0: level 2, at (3, 2) - adjacent to food (west)
# Agent 1: level 1, at (2, 3) - adjacent to food (north)
# Agent 2: level 1, at (4, 4) - NOT adjacent
agent_positions = jnp.array([
    [3, 2],  # Agent 0
    [2, 3],  # Agent 1
    [4, 4],  # Agent 2
])
agent_levels = jnp.array([2, 1, 1])

# Create manual state
from socialjax.environments.lb_foraging.lb_foraging import State
manual_state = State(
    agent_positions=agent_positions,
    agent_levels=agent_levels,
    food_grid=food_grid,
    inner_t=0,
    outer_t=0,
)

for i in range(3):
    print(f"  Agent {i}: pos={agent_positions[i]}, level={agent_levels[i]}")
print(f"\nFood: pos=(3, 3), level=2")
print(f"Food requires: total_level >= 2")

# Test Case 1: Both adjacent agents execute LOAD (should succeed)
print("\n" + "-" * 60)
print("[Case 1] Agent 0 (L2) and Agent 1 (L1) both LOAD")
print("Expected: Success! Total level = 2+1 = 3 >= 2")
print("Expected rewards (with normalization):")
print("  Agent 0: (2 × 2) / 3 = 1.333")
print("  Agent 1: (1 × 2) / 3 = 0.667")
print("  Agent 2: 0 (not participating)")

actions = jnp.array([5, 5, 0])  # LOAD, LOAD, NONE
key, subkey = jax.random.split(key)
obs, state, rewards, dones, info = env.step_env(subkey, manual_state, actions)

print("\nActual rewards:", rewards)
print("Food remaining:", int(jnp.sum(state.food_grid > 0)))
assert int(jnp.sum(state.food_grid > 0)) == 0, "Food should be collected!"
assert abs(rewards[0] - 1.333) < 0.01, f"Agent 0 reward wrong: {rewards[0]}"
assert abs(rewards[1] - 0.667) < 0.01, f"Agent 1 reward wrong: {rewards[1]}"
assert abs(rewards[2] - 0.0) < 0.01, f"Agent 2 reward wrong: {rewards[2]}"
print("[OK] Test Case 1 PASSED")

# Reset for Test Case 2
food_grid = jnp.zeros(env.grid_shape, dtype=jnp.int32)
food_grid = food_grid.at[3, 3].set(12)  # level 2 food
manual_state = State(
    agent_positions=agent_positions,
    agent_levels=agent_levels,
    food_grid=food_grid,
    inner_t=0,
    outer_t=0,
)

print("\n" + "-" * 60)
print("[Case 2] Only Agent 1 (L1) executes LOAD")
print("Expected: FAIL! Total level = 1 < 2")
print("Expected rewards:")
print("  Agent 0: 0 (not participating)")
print("  Agent 1: -0.1 (penalty for failed LOAD)")
print("  Agent 2: 0 (not participating)")

actions = jnp.array([0, 5, 0])  # NONE, LOAD, NONE
key, subkey = jax.random.split(key)
obs, state, rewards, dones, info = env.step_env(subkey, manual_state, actions)

print("\nActual rewards:", rewards)
print("Food remaining:", int(jnp.sum(state.food_grid > 0)))
assert int(jnp.sum(state.food_grid > 0)) == 1, "Food should NOT be collected!"
assert abs(rewards[0] - 0.0) < 0.01, f"Agent 0 reward wrong: {rewards[0]}"
assert abs(rewards[1] - (-0.1)) < 0.01, f"Agent 1 reward wrong: {rewards[1]}"
assert abs(rewards[2] - 0.0) < 0.01, f"Agent 2 reward wrong: {rewards[2]}"
print("[OK] Test Case 2 PASSED")

# Test Case 3: Non-adjacent agent tries LOAD
food_grid = jnp.zeros(env.grid_shape, dtype=jnp.int32)
food_grid = food_grid.at[3, 3].set(12)  # level 2 food
manual_state = State(
    agent_positions=agent_positions,
    agent_levels=agent_levels,
    food_grid=food_grid,
    inner_t=0,
    outer_t=0,
)

print("\n" + "-" * 60)
print("[Case 3] Only Agent 2 (not adjacent) executes LOAD")
print("Expected: No penalty (not adjacent to any food)")
print("Expected rewards: all 0")

actions = jnp.array([0, 0, 5])  # NONE, NONE, LOAD
key, subkey = jax.random.split(key)
obs, state, rewards, dones, info = env.step_env(subkey, manual_state, actions)

print("\nActual rewards:", rewards)
print("Food remaining:", int(jnp.sum(state.food_grid > 0)))
assert int(jnp.sum(state.food_grid > 0)) == 1, "Food should NOT be collected!"
assert abs(rewards[0] - 0.0) < 0.01, f"Agent 0 reward wrong: {rewards[0]}"
assert abs(rewards[1] - 0.0) < 0.01, f"Agent 1 reward wrong: {rewards[1]}"
assert abs(rewards[2] - 0.0) < 0.01, f"Agent 2 reward wrong: {rewards[2]}"
print("[OK] Test Case 3 PASSED")

# Test Case 4: Test WITHOUT normalization
print("\n" + "-" * 60)
print("[Case 4] Test without normalization")
env_no_norm = socialjax.make(
    "lb_foraging",
    num_agents=3,
    num_food=1,
    max_food_level=3,
    max_player_level=2,
    num_inner_steps=10,
    sight=7,
    jit=False,
    cnn=True,
    normalize_reward=False,  # NO normalization
    load_penalty=0.1,
)

food_grid = jnp.zeros(env_no_norm.grid_shape, dtype=jnp.int32)
food_grid = food_grid.at[3, 3].set(12)  # level 2 food
manual_state = State(
    agent_positions=agent_positions,
    agent_levels=agent_levels,
    food_grid=food_grid,
    inner_t=0,
    outer_t=0,
)

print("Agent 0 (L2) and Agent 1 (L1) both LOAD")
print("Expected rewards (NO normalization):")
print("  Agent 0: 2 × 2 = 4.0")
print("  Agent 1: 1 × 2 = 2.0")
print("  Agent 2: 0")

actions = jnp.array([5, 5, 0])  # LOAD, LOAD, NONE
key, subkey = jax.random.split(key)
obs, state, rewards, dones, info = env_no_norm.step_env(subkey, manual_state, actions)

print("\nActual rewards:", rewards)
assert abs(rewards[0] - 4.0) < 0.01, f"Agent 0 reward wrong: {rewards[0]}"
assert abs(rewards[1] - 2.0) < 0.01, f"Agent 1 reward wrong: {rewards[1]}"
assert abs(rewards[2] - 0.0) < 0.01, f"Agent 2 reward wrong: {rewards[2]}"
print("[OK] Test Case 4 PASSED")

print("\n" + "=" * 60)
print("*** ALL REWARD TESTS PASSED! ***")
print("=" * 60)
print("\nReward calculation logic is CORRECT:")
print("1. [OK] Success: reward_i = (agent_i.level x food.level) / total_participating_levels")
print("2. [OK] Failure (adjacent + LOAD but insufficient level): -0.1 penalty")
print("3. [OK] No penalty if not adjacent to food")
print("4. [OK] Normalization works correctly")
