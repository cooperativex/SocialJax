from pathlib import Path

import jax
import numpy as onp
from PIL import Image

from socialjax import make

ORIENT_NAME = {
    0: "Up",
    1: "Right",
    2: "Down",
    3: "Left"
}

def print_agent_info(agent_index, agent_locs, agent_colors, obs, verbose):
    if not verbose:
        return

    row, col, orient = agent_locs[agent_index]
    orient_str = ORIENT_NAME.get(int(orient), f"Unknown({orient})")
    color_str = "(unknown)"
    if agent_colors is not None and agent_index < len(agent_colors):
        color = agent_colors[agent_index]
        color_str = f"RGB{color}"

    print(f"Agent {agent_index}:")
    print(f"   - Location: (row={row}, col={col})")
    print(f"   - Orientation: {orient} ({orient_str})")
    print(f"   - Color: {color_str}")

def main():
    """
    Creates a one-off scenario to debug iron ore mining.
    1) Places iron ore manually next to agent.
    2) Steps through a fixed sequence of actions to test.
    """
    verbose = True
    num_agents = 2
    num_inner_steps = 6
    num_outer_steps = 1

    rng = jax.random.PRNGKey(123)

    # Create environment
    env = make(
        "coop_mining",
        num_inner_steps=num_inner_steps,
        num_outer_steps=num_outer_steps,
        num_agents=num_agents,
        seed=123,
    )

    # Directory for saving images (optional)
    root_dir = f"tests/coop_mining_manual_debug_ore"
    path = Path(root_dir + "/state_pics")
    path.mkdir(parents=True, exist_ok=True)

    # Reset the environment
    rng, subkey = jax.random.split(rng)
    obs, old_state = env.reset(subkey)

    # Manually place an iron ore next to each agent for debugging
    # For example, place iron ore to the right (col + 1) of each agent (if valid).
    state_np = onp.array(old_state.grid, copy=True)  # CPU copy

    agent_locs_np = onp.array(old_state.agent_locs)
    for i in range(num_agents):
        ar, ac, _orient = agent_locs_np[i]
        # Make the agents face up
        agent_locs_np[i] = [ar, ac, 0]
        # We'll place an iron_ore if col+1 is not out of bounds
        if ac + 1 < state_np.shape[1]:
            state_np[ar, ac + 1] = 4  # Items.iron_ore = 4
            state_np[ar, ac + 2] = 4  # Items.iron_ore = 4

    # Re-inject updated grid into the JAX state
    new_grid = jax.device_put(state_np)
    new_agent_locs = jax.device_put(agent_locs_np)
    old_state = old_state.replace(grid=new_grid)
    old_state = old_state.replace(agent_locs=new_agent_locs)

    # Optionally, re-render the initial scenario with our manually placed ore
    first_img = env.render(old_state)
    Image.fromarray(first_img).save(f"{root_dir}/state_pics/manual_init.png")

    # The environment's Actions are (by IntEnum):
    #   turn_left=0, turn_right=1, step_left=2, step_right=3,
    #   forward=4, backward=5, stay=6, mine=7
    # For now, agent1 always does "stay"

    agent0_actions = [7, 1, 7, 6, 7, 3]
    agent1_actions = [6, 6, 6, 6, 6, 6]  # just chill for now

    # Let's step through these manually:
    for t in range(num_inner_steps):
        # Action for each agent
        actions = [agent0_actions[t], agent1_actions[t]]

        obs, state, reward, done, info = env.step_env(rng, old_state, actions)

        # Print debug info
        print("="*30)
        print(f"Time step {t}: actions={actions}")
        print("Rewards:", reward)
        for i in range(num_agents):
            row, col, orient = onp.array(state.agent_locs[i])
            print(f"Agent {i}: row={row}, col={col}, orientation={orient}  => reward={reward[i]:.2f}")

        # Render & save
        img = env.render(state)
        Image.fromarray(img).save(f"{root_dir}/state_pics/state_{t}.png")

        old_state = state

    print("Done with manual scenario.")


if __name__ == "__main__":
    main()
