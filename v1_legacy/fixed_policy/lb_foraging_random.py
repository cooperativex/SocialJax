"""
Generate GIF visualization of LB Foraging environment with random policy
"""
import sys
import os
from pathlib import Path

# Add parent directory to path for socialjax import
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from socialjax import make


def main():
    # Test configuration
    config = {
        'num_agents': 3,
        'num_food': 5,
        'max_food_level': 3,
        'max_player_level': 2,
        'num_inner_steps': 50,
        'force_coop': False,
        'sight': 7,
    }

    verbose = True

    num_agents = config['num_agents']
    num_inner_steps = config['num_inner_steps']
    num_food = config['num_food']
    force_coop = config['force_coop']

    # Random seed
    rng = jax.random.PRNGKey(42)

    # Create environment
    env = make(
        "lb_foraging",
        num_agents=num_agents,
        num_food=num_food,
        max_food_level=config['max_food_level'],
        max_player_level=config['max_player_level'],
        num_inner_steps=num_inner_steps,
        force_coop=force_coop,
        sight=config['sight'],
        jit=False,  # Disable JIT for easier debugging
        cnn=True,
        normalize_reward=True,
        load_penalty=0.0,  # No penalty for failed LOAD
    )

    # Prepare output directory
    coop_str = "coop" if force_coop else "individual"
    grid_size = env.grid_shape[0]
    root_dir = f"tests/lb_foraging_a{num_agents}_g{grid_size}_f{num_food}_{coop_str}"
    path = Path(root_dir + "/state_pics")
    path.mkdir(parents=True, exist_ok=True)

    # Reset environment
    rng, subkey = jax.random.split(rng)
    obs, state = env.reset(subkey)

    if verbose:
        print("\n" + "=" * 60)
        print(f"LB Foraging Environment - Random Policy")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  - Agents: {num_agents}")
        print(f"  - Grid Size: {env.grid_shape[0]}x{env.grid_shape[1]}")
        print(f"  - Food Items: {num_food}")
        print(f"  - Max Food Level: {config['max_food_level']}")
        print(f"  - Max Player Level: {config['max_player_level']}")
        print(f"  - Force Cooperation: {force_coop}")
        print(f"  - Episode Length: {num_inner_steps}")
        print("\nInitial State:")
        print(f"  - Agent Positions: {np.array(state.agent_positions)}")
        print(f"  - Agent Levels: {np.array(state.agent_levels)}")
        food_count = int(jnp.sum(state.food_grid > 0))
        print(f"  - Active Food: {food_count}/{num_food}")
        print("=" * 60 + "\n")

    # Store frames for GIF
    pics = []

    # Track statistics
    total_reward = 0.0
    food_collected = 0
    # Track cumulative rewards for each agent
    cumulative_rewards = {str(i): 0.0 for i in range(num_agents)}

    # Render and save initial state (with zero rewards)
    img = env.render(state, cell_size=50, cumulative_rewards=cumulative_rewards)
    Image.fromarray(img).save(f"{root_dir}/state_pics/init_state.png")
    pics.append(img)

    # Run episode with random actions
    for t in range(num_inner_steps):
        # Generate random actions for each agent (0-5: NONE, N, S, W, E, LOAD)
        rng, subkey = jax.random.split(rng)
        actions = jax.random.randint(subkey, shape=(num_agents,), minval=0, maxval=6)

        # Step environment
        rng, subkey = jax.random.split(rng)
        obs, state, rewards, dones, info = env.step_env(
            subkey, state, actions
        )

        # Update statistics and cumulative rewards
        step_reward = float(jnp.sum(rewards))
        total_reward += step_reward

        # Update cumulative rewards for each agent
        for i in range(num_agents):
            cumulative_rewards[str(i)] += float(rewards[i])

        if verbose and step_reward > 0:
            print(f"Step {t + 1}:")
            print(f"  - Actions: {[int(a) for a in actions]}")
            print(f"  - Rewards: {[float(rewards[i]) for i in range(num_agents)]}")
            print(f"  - Cumulative Rewards: {[cumulative_rewards[str(i)] for i in range(num_agents)]}")
            print(f"  - Total Reward This Step: {step_reward:.2f}")
            food_count = int(jnp.sum(state.food_grid > 0))
            print(f"  - Active Food: {food_count}/{num_food}")

        # Render and save with cumulative rewards
        img = env.render(state, cell_size=50, cumulative_rewards=cumulative_rewards)
        Image.fromarray(img).save(f"{root_dir}/state_pics/state_{t + 1}.png")
        pics.append(img)

        # Check if episode done
        if dones["__all__"]:
            if verbose:
                print(f"\nEpisode ended at step {t + 1}")
                food_remaining = int(jnp.sum(state.food_grid > 0))
                print(f"Reason: {'All food collected' if food_remaining == 0 else 'Time limit reached'}")
            break

    # Final statistics
    food_remaining = int(jnp.sum(state.food_grid > 0))
    food_collected = num_food - food_remaining

    if verbose:
        print("\n" + "=" * 60)
        print("Episode Summary:")
        print(f"  - Total Steps: {int(state.inner_t)}")
        print(f"  - Total Reward: {total_reward:.2f}")
        print(f"  - Food Collected: {food_collected}/{num_food}")
        print(f"  - Collection Rate: {food_collected / num_food * 100:.1f}%")
        print("\nAgent Cumulative Rewards:")
        for i in range(num_agents):
            print(f"  - Agent {i}: {cumulative_rewards[str(i)]:.4f}")
        print("=" * 60 + "\n")

    # Create and save GIF
    print(f"Saving GIF to {root_dir}/lb_foraging_random.gif")
    frames = [Image.fromarray(im) for im in pics]
    frames[0].save(
        f"{root_dir}/lb_foraging_random.gif",
        format="GIF",
        save_all=True,
        optimize=False,
        append_images=frames[1:],
        duration=300,  # 300ms per frame
        loop=0,
    )

    print(f"GIF saved successfully!")
    print(f"Individual frames saved to: {root_dir}/state_pics/")


if __name__ == "__main__":
    main()
