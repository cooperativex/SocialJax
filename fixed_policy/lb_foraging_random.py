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
import numpy as np
from PIL import Image

from socialjax import make


def main():
    # Test configurations
    configs = [
        {
            'num_agents': 3,
            'grid_size': 8,
            'num_food': 5,
            'max_food_level': 3,
            'max_agent_level': 3,
            'num_inner_steps': 50,
            'force_coop': False,
        },
    ]

    verbose = True

    for config in configs:
        num_agents = config['num_agents']
        num_inner_steps = config['num_inner_steps']
        grid_size = config['grid_size']
        num_food = config['num_food']
        force_coop = config['force_coop']

        # Random seed
        rng = jax.random.PRNGKey(42)

        # Create environment
        env = make(
            "lb_foraging",
            num_agents=num_agents,
            grid_size=grid_size,
            num_food=num_food,
            max_food_level=config['max_food_level'],
            max_agent_level=config['max_agent_level'],
            num_inner_steps=num_inner_steps,
            force_coop=force_coop,
            sight=3,
            jit=False,  # Disable JIT for easier debugging
            cnn=True,
        )

        # Prepare output directory
        coop_str = "coop" if force_coop else "individual"
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
            print(f"  - Grid Size: {grid_size}x{grid_size}")
            print(f"  - Food Items: {num_food}")
            print(f"  - Max Food Level: {config['max_food_level']}")
            print(f"  - Max Agent Level: {config['max_agent_level']}")
            print(f"  - Force Cooperation: {force_coop}")
            print(f"  - Episode Length: {num_inner_steps}")
            print("\nInitial State:")
            print(f"  - Agent Levels: {np.array(state.agent_levels)}")
            print(f"  - Food Levels: {np.array(state.food_levels)}")
            print(f"  - Active Food: {np.sum(state.food_active)}/{num_food}")
            print("=" * 60 + "\n")

        # Store frames for GIF
        pics = []

        # Render and save initial state
        img = env.render(state, cell_size=50)
        Image.fromarray(img).save(f"{root_dir}/state_pics/init_state.png")
        pics.append(img)

        # Track statistics
        total_reward = 0.0
        food_collected = 0

        # Run episode with random actions
        for t in range(num_inner_steps):
            # Generate random actions for each agent
            rng, *rngs = jax.random.split(rng, num_agents + 1)
            actions = [
                int(jax.random.randint(rngs[a], shape=(), minval=0, maxval=env.action_space(a).n))
                for a in range(num_agents)
            ]

            # Step environment
            obs, state, rewards, dones, info = env.step_env(
                rng, state, actions
            )

            # Update statistics
            step_reward = sum([float(rewards[str(i)]) for i in range(num_agents)])
            total_reward += step_reward

            if verbose and step_reward > 0:
                print(f"Step {t + 1}:")
                print(f"  - Actions: {actions}")
                print(f"  - Rewards: {[float(rewards[str(i)]) for i in range(num_agents)]}")
                print(f"  - Total Reward This Step: {step_reward:.2f}")
                print(f"  - Active Food: {np.sum(state.food_active)}/{num_food}")

            # Render and save
            img = env.render(state, cell_size=50)
            Image.fromarray(img).save(f"{root_dir}/state_pics/state_{t + 1}.png")
            pics.append(img)

            # Check if episode done
            if dones["__all__"]:
                if verbose:
                    print(f"\nEpisode ended at step {t + 1}")
                    print(f"Reason: {'All food collected' if np.all(~state.food_active) else 'Time limit reached'}")
                break

        # Final statistics
        food_collected = num_food - np.sum(state.food_active)

        if verbose:
            print("\n" + "=" * 60)
            print("Episode Summary:")
            print(f"  - Total Steps: {int(state.step_count)}")
            print(f"  - Total Reward: {total_reward:.2f}")
            print(f"  - Food Collected: {food_collected}/{num_food}")
            print(f"  - Collection Rate: {food_collected / num_food * 100:.1f}%")
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
