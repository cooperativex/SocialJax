from pathlib import Path

import jax
import numpy as onp
from PIL import Image

from socialjax import make
from socialjax.environments.coop_mining.coop_mining import Items

ORIENT_NAME = {
    0: "Up",
    1: "Right",
    2: "Down",
    3: "Left"
}


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


def print_agent_info(agent_index, agent_locs, agent_colors, obs, verbose, print_partial_view):
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

    # Partial observation from the `obs` dictionary
    final_obs_for_agent = onp.array(obs[agent_index])  # (H, W, len(Items)+5)
    item_layer = extract_item_layer(final_obs_for_agent)
    partial_obs_np = onp.array(item_layer)  # shape (h, w)

    print(f"Agent {agent_index}:")
    print(f"   - Location: (row={row}, col={col})")
    print(f"   - Orientation: {orient} ({orient_str})")
    print(f"   - Color: {color_str}")
    if print_partial_view:
        print(f"   - Partial View:\n{partial_obs_np}")


def print_initial_state(num_agents, agent_locs, agent_colors, obs, verbose, print_partial_view):
    if not verbose:
        return

    print("\n=== Initial Agent States ===")
    for i in range(num_agents):
        print_agent_info(i, agent_locs, agent_colors, obs, verbose, print_partial_view)
    print("===== End Debug Print =====\n")


def print_timestep_info(t, actions, reward, num_agents, agent_locs, agent_colors, obs, verbose, print_partial_view):
    if not verbose:
        return

    print("###################")
    print(f"timestep: {t} -> {t + 1}")
    print(f"actions: {actions}")
    print(f"rewards: {reward}")

    for i in range(num_agents):
        print_agent_info(i, agent_locs, agent_colors, obs, verbose, print_partial_view)
    print("###################")


def main():
    # Example: test different agent population sizes
    agent_pop_sizes = [3]
    verbose = True  # Set to False to disable printing
    print_partial_view = False

    for n_a in agent_pop_sizes:
        # Set environment parameters
        num_agents = n_a
        num_inner_steps = 15  # how many steps per episode
        num_outer_steps = 1
        rng = jax.random.PRNGKey(123)

        # Create environment
        env = make(
            "coop_mining",
            num_inner_steps=num_inner_steps,
            num_outer_steps=num_outer_steps,
            num_agents=num_agents,
            jit=False,
        )

        # Prepare a directory for saving images
        root_dir = f"tests/coop_mining_a{num_agents}_i{num_inner_steps}_o{num_outer_steps}"
        path = Path(root_dir + "/state_pics")
        path.mkdir(parents=True, exist_ok=True)

        for o_t in range(num_outer_steps):
            # Reset environment
            rng, subkey = jax.random.split(rng)
            obs, old_state = env.reset(subkey)

            agent_locs = onp.array(old_state.agent_locs)
            agent_colors = getattr(env, "_agent_colors", None)

            # Print initial state if verbose
            print_initial_state(num_agents, agent_locs, agent_colors, obs, verbose, print_partial_view)

            # We'll store all frames in `pics` for a GIF
            pics = []

            # Render and save the initial state
            img = env.render(old_state)
            Image.fromarray(img).save(f"{root_dir}/state_pics/init_state.png")
            pics.append(img)

            # Step through the inner episode
            for t in range(num_inner_steps):
                rng, *rngs = jax.random.split(rng, num_agents + 1)
                actions = [
                    jax.random.randint(rngs[a], shape=(), minval=0, maxval=env.action_space(a).n)
                    for a in range(num_agents)
                ]

                # Step the environment
                obs, state, reward, done, info = env.step_env(
                    rng, old_state, [a for a in actions]
                )

                # Print timestep info if verbose
                print_timestep_info(t, actions, reward, num_agents, agent_locs, agent_colors, obs, verbose, print_partial_view)

                # Render & save
                img = env.render(state)
                Image.fromarray(img).save(f"{root_dir}/state_pics/state_{t + 1}.png")
                pics.append(img)

                old_state = state

            # Create and save a GIF of the entire inner episode
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
