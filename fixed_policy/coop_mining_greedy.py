import random
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as onp
from PIL import Image

from socialjax import make
from socialjax.environments.coop_mining.coop_mining import Actions, ViewConfig
from socialjax.environments.coop_mining.coop_mining import Items
from socialjax.environments.coop_mining.coop_mining import State

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

    # Orientation
    orient_str = ORIENT_NAME.get(int(orient), f"Unknown({orient})")

    # If the environment has color info
    color_str = "(unknown)"
    if agent_colors is not None and agent_index < len(agent_colors):
        color = agent_colors[agent_index]
        color_str = f"RGB{color}"

    print(f"Agent {agent_index}:")
    print(f"   - Location: (row={row}, col={col})")
    print(f"   - Orientation: {orient} ({orient_str})")
    print(f"   - Color: {color_str}")


def print_initial_state(num_agents, agent_locs, agent_colors, obs, verbose):
    if not verbose:
        return

    print("\n=== Initial Agent States ===")
    for i in range(num_agents):
        print_agent_info(i, agent_locs, agent_colors, obs, verbose)
    print("===== End Debug Print =====\n")


def print_timestep_info(t, actions, reward, num_agents, agent_locs, agent_colors, obs, verbose):
    if not verbose:
        return

    print("###################")
    print(f"timestep: {t} -> {t + 1}")
    print(f"actions: {actions}")
    print(f"rewards: {reward}")

    for i in range(num_agents):
        print_agent_info(i, agent_locs, agent_colors, obs, verbose)
    print("###################")


# We'll keep one global target (row, col) per agent: None if no current target.
# Replace this with a class/structure if you prefer.
agent_targets = []


def init_targets(num_agents):
    global agent_targets
    agent_targets = [None] * num_agents


def local_to_global_cpu(agent_row, agent_col, orient, local_row, local_col, view_cfg):
    """
    CPU version of local->global from your env, using 0=Up,1=Right,2=Down,3=Left.
    """
    # in local coords, the agent is at (view_cfg.backward, view_cfg.left).
    agent_local_r = view_cfg.backward
    agent_local_c = view_cfg.left
    dr_local = local_row - agent_local_r
    dc_local = local_col - agent_local_c

    if orient == 0:  # Up
        gr = agent_row - dr_local
        gc = agent_col - dc_local
    elif orient == 1:  # Right
        gr = agent_row - dc_local
        gc = agent_col + dr_local
    elif orient == 2:  # Down
        gr = agent_row + dr_local
        gc = agent_col + dc_local
    else:  # Left
        gr = agent_row + dc_local
        gc = agent_col - dr_local
    return gr, gc


def extract_item_layer(final_obs_for_agent):
    """
    Given a single agent's final obs array of shape (H, W, len(Items) + 5),
    where the first (len(Items)-1) channels are one-hot for items,
    compute a 2D array (H, W) of raw item IDs in [0..N].

    Example: if 0 => Items.empty, 1 => Items.wall, 2 => ore_wait, 3 => spawn_point, etc.
    Adjust the +1 offset if needed to match your Items() enum.
    """
    # final_obs_for_agent: shape (H, W, len(Items)+5)
    H, W, depth = final_obs_for_agent.shape
    num_item_ch = (len(Items) - 1)  # e.g. 6 in your case

    # 1) slice out the item channels [0 .. num_item_ch)
    item_oh = final_obs_for_agent[..., :num_item_ch]  # shape = (H, W, num_item_ch)

    # 2) argmax along 'channel' to get the index in [0..(num_item_ch-1)]
    #    e.g. 0 might represent "Items.empty - 1" if you used (grid-1).
    item_idx = onp.argmax(item_oh, axis=-1)  # shape = (H, W)

    # 3) SHIFT +1 IF your one-hot was built via (grid-1).
    #    In your code, you do:  grids_1hot = jax.nn.one_hot(grids - 1, ...)
    #    so item_idx==0 => (grids-1) ==0 => grids==1 => Items.wall? or empty?
    #    You need to see how you want to interpret that. For example:
    item_id = item_idx + 1  # shape = (H, W)

    # Now item_id==1 means "Items.wall" if your Items enum says 1 => wall.
    # Or you might do: item_id = item_idx, if you prefer 0 => wall, etc.
    # Just confirm it matches your Items(...) numbering.

    return item_id  # shape (H, W) of raw item IDs


def find_closest_ore_in_local(obs, center_r, center_c):
    """
    Returns (local_r, local_c) of the nearest ore, or None if none visible.
    """
    ore_positions = onp.argwhere(
        (obs == Items.iron_ore) |
        (obs == Items.gold_ore) |
        (obs == Items.gold_partial)
    )
    if ore_positions.size == 0:
        return None
    dists = onp.sum(onp.abs(ore_positions - (center_r, center_c)), axis=1)
    idx = onp.argmin(dists)
    return tuple(ore_positions[idx])  # (local_r, local_c)


def compute_action_for_target(agent_row, agent_col, orient, target_r, target_c):
    dr = target_r - agent_row
    dc = target_c - agent_col

    # Orientation to vector
    orient_map = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    forward_vec = orient_map[orient]

    # If target is aligned in front
    aligned = (dr == 0 and forward_vec[0] == 0 and dc * forward_vec[1] > 0) or \
              (dc == 0 and forward_vec[1] == 0 and dr * forward_vec[0] > 0)

    # Distance in that direction
    dist = abs(dr) + abs(dc)

    # If aligned and within 3, mine
    if aligned and dist <= 3:
        return Actions.mine

    # Randomly decide whether to prioritize row or column movement
    prioritize_row = random.choice([True, False])

    if prioritize_row:
        if dr != 0:  # Prioritize vertical movement
            desired_orient = 0 if dr < 0 else 2
        else:  # If no vertical movement needed, adjust horizontally
            desired_orient = 3 if dc < 0 else 1
    else:
        if dc != 0:  # Prioritize horizontal movement
            desired_orient = 3 if dc < 0 else 1
        else:  # If no horizontal movement needed, adjust vertically
            desired_orient = 0 if dr < 0 else 2

    if desired_orient != orient:
        left_turn = (orient - 1) % 4
        return Actions.turn_left if left_turn == desired_orient else Actions.turn_right

    # Move forward/side/back to get aligned
    if desired_orient == 0:
        return Actions.forward
    elif desired_orient == 1:
        return Actions.step_right
    elif desired_orient == 2:
        return Actions.backward
    else:
        return Actions.step_left


def greedy_policy(obs, state, env):
    """
    For each agent:
      1) If agent_targets[i] is None, find a new nearest ore in local obs (if any),
         convert to global coords, store in agent_targets[i].
      2) If there is a stored target, check if that cell is still ore.
         If not ore anymore, set agent_targets[i] = None.
      3) Otherwise, move/turn/mine toward that stored target.
      4) If no target found, random explore.
    """
    global agent_targets
    actions = []

    # Precompute center of local obs
    h = env.view_config.forward + env.view_config.backward + 1
    w = env.view_config.left + env.view_config.right + 1
    center_r = env.view_config.backward
    center_c = env.view_config.left

    # Convert agent_locs to numpy
    agent_locs_np = onp.array(state.agent_locs)

    # We'll need the global grid to confirm if our target is still ore
    grid_np = onp.array(state.grid)

    for i in range(env.num_agents):
        ar, ac, orient = agent_locs_np[i]
        orient = int(orient)  # Make sure it's a regular int

        # If no target, pick one
        if agent_targets[i] is None:
            # Check local obs
            final_obs_for_agent = onp.array(obs[i])  # (H, W, len(Items)+5)
            item_layer = extract_item_layer(final_obs_for_agent)
            local_view = onp.array(item_layer)  # shape (h, w)
            closest_local = find_closest_ore_in_local(local_view, center_r, center_c)
            if closest_local is not None:
                lr, lc = closest_local
                # Convert to global
                tr, tc = local_to_global_cpu(ar, ac, orient, lr, lc, env.view_config)
                agent_targets[i] = (tr, tc)

        # If we still have no target, random explore
        if agent_targets[i] is None:
            actions.append(random.choice([
                Actions.turn_left, Actions.turn_right,
                Actions.forward, Actions.step_left,
                Actions.step_right, Actions.stay
            ]))
            continue

        # Otherwise we have a target => check if it's still ore
        target_r, target_c = agent_targets[i]
        # Safety check bounds
        if not (0 <= target_r < grid_np.shape[0] and 0 <= target_c < grid_np.shape[1]):
            # target is out of bounds => forget it
            agent_targets[i] = None
            # do a random action
            actions.append(random.choice([
                Actions.turn_left, Actions.turn_right,
                Actions.forward, Actions.step_left,
                Actions.step_right, Actions.stay
            ]))
            continue

        item_here = grid_np[target_r, target_c]
        if item_here not in [Items.iron_ore, Items.gold_ore, Items.gold_partial]:
            # It's no longer ore => forget this target
            agent_targets[i] = None
            # random action or pick new target next time
            actions.append(random.choice([
                Actions.turn_left, Actions.turn_right,
                Actions.forward, Actions.step_left,
                Actions.step_right, Actions.stay
            ]))
            continue

        # If we get here, we do have a valid ore target => act to approach/mine it
        act = compute_action_for_target(ar, ac, orient, target_r, target_c)
        actions.append(act)

    return actions


def add_fixed_ore(grid, fixed_ore_positions, ore_type):
    """Adds fixed ore locations to the grid."""
    for pos in fixed_ore_positions:
        row, col = pos
        if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]:
            grid[row, col] = ore_type
    return grid


def main():
    # Example: test different agent population sizes
    agent_pop_sizes = [7]
    verbose = True
    render = True
    ore_type = Items.iron_ore

    for n_a in agent_pop_sizes:
        # Set environment parameters
        num_agents = n_a
        num_inner_steps = 10  # how many steps per episode
        num_outer_steps = 1
        rng = jax.random.PRNGKey(123)

        # Create environment
        env = make(
            "coop_mining",
            num_inner_steps=num_inner_steps,
            num_outer_steps=num_outer_steps,
            num_agents=num_agents,
            jit=False,
            view_config=ViewConfig(forward=5, backward=1, left=3, right=3),
        )

        # Prepare a directory for saving images
        root_dir = f"tests/coop_mining_greedy_a{num_agents}_i{num_inner_steps}_o{num_outer_steps}"
        path = Path(root_dir + "/state_pics")
        path.mkdir(parents=True, exist_ok=True)

        for o_t in range(num_outer_steps):
            # Reset environment
            rng, subkey = jax.random.split(rng)
            obs, old_state = env.reset(subkey)

            init_targets(env.num_agents)  # Initialize targets for all agents

            # Define fixed ore positions
            fixed_ore_positions = jnp.array([
                [3, 3], [3, 8], [6, 6], [7, 2], [2, 7],
                [3, 20], [6, 18], [8, 25], [5, 22], [2, 24],
                [20, 3], [22, 5], [25, 7], [18, 6], [23, 2],
                [22, 22], [25, 25], [20, 23], [23, 20], [18, 18],
                [13, 13], [12, 15], [15, 12], [14, 16], [16, 14],
                [10, 8], [11, 9], [9, 10], [8, 15], [7, 13],
                [19, 21], [17, 19], [18, 14], [21, 19], [20, 25],
                [4, 18], [5, 21], [6, 20], [3, 23], [8, 24],
                [25, 4], [24, 6], [23, 8], [21, 5], [20, 2],
                [14, 8], [13, 7], [15, 6], [12, 5], [11, 4],
            ], dtype=jnp.int16)

            # Modify the grid to add fixed ore
            state_np = onp.array(old_state.grid, copy=True)  # Convert to CPU for modification
            state_np = add_fixed_ore(state_np, fixed_ore_positions, ore_type)

            # Update the environment state
            new_grid = jax.device_put(state_np)
            old_state = old_state.replace(grid=new_grid)

            agent_locs = onp.array(old_state.agent_locs)
            agent_colors = getattr(env, "_agent_colors", None)

            # Print initial state if verbose
            print_initial_state(num_agents, agent_locs, agent_colors, obs, verbose)

            # We'll store all frames in `pics` for a GIF
            pics = []

            # Render and save the initial state
            if render:
                img = env.render(old_state)
                Image.fromarray(img).save(f"{root_dir}/state_pics/init_state.png")
                pics.append(img)

            # Step through the inner episode
            for t in range(num_inner_steps):
                # Generate random actions for agents (to be used when no ore is visible)
                rng, *rngs = jax.random.split(rng, num_agents + 1)

                # Determine actions using the greedy policy
                actions = greedy_policy(obs, old_state, env)

                # Step the environment
                obs, state, reward, done, info = env.step_env(
                    rng, old_state, actions
                )

                # Print timestep info if verbose
                print_timestep_info(t, actions, reward, num_agents, state.agent_locs, agent_colors, obs, verbose)

                # Render & save
                if render:
                    img = env.render(state)
                    Image.fromarray(img).save(f"{root_dir}/state_pics/state_{t + 1}.png")
                    pics.append(img)

                old_state = state

            # Create and save a GIF of the entire inner episode
            if render:
                print("Saving GIF for outer_t =", o_t)
                frames = [Image.fromarray(im) for im in pics]
                frames[0].save(
                    f"{root_dir}/state_outer_step_{o_t + 1}.gif",
                    format="GIF",
                    save_all=True,
                    optimize=False,
                    append_images=frames[1:],
                    duration=200,
                    loop=0,
                )


if __name__ == "__main__":
    main()
