"""
Shared evaluation utilities for all algorithms.

This module provides evaluation functions for trained MARL policies.
Different algorithms have slightly different evaluation patterns:
- IPPO: Uses ActorCritic with PARAMETER_SHARING logic + WandB GIF logging
- MAPPO/IRAT: Use Actor network (no critic in eval) without WandB logging
- SVO/TRANSFER: Use ActorCritic without PARAMETER_SHARING logic, no WandB logging
"""

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from pathlib import Path
import wandb

from algorithms.utils.networks import ActorCritic, SmallActor
from algorithms.utils.data_utils import unbatchify


def evaluate_ippo(params, env, save_path, config):
    """
    Evaluation function for IPPO algorithm.

    Supports both parameter sharing and individual networks.
    Logs evaluation GIF to WandB.

    Args:
        params: Model parameters (single or list for multi-agent)
        env: Environment instance
        save_path: Path where params were saved (unused but kept for API compatibility)
        config: Configuration dictionary
    """
    rng = jax.random.PRNGKey(0)

    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng)
    done = False

    pics = []
    img = env.render(state)
    pics.append(img)

    # Extract environment name for root_dir
    env_name = config["ENV_NAME"]
    # Map environment names to evaluation directories
    env_dir_mapping = {
        "clean_up": "cleanup",
        "coin_game": "coins",
        "coop_mining": "coop_mining",
        "gift": "gift",
        "harvest_common_open": "harvest_common",
        "harvest_common_closed": "harvest_closed",
        "harvest_common_partnership": "harvest_partnership",
        "mushrooms": "mushrooms",
        "pd_arena": "pd_arena",
        "territory_open": "territory_open",
    }
    env_dir = env_dir_mapping.get(env_name, env_name)
    root_dir = f"evaluation/{env_dir}"
    path = Path(root_dir + "/state_pics")
    path.mkdir(parents=True, exist_ok=True)

    for o_t in range(config["GIF_NUM_FRAMES"]):
        # Use model to select actions
        if config.get("PARAMETER_SHARING", True):
            obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *env.observation_space()[0].shape)
            network = ActorCritic(action_dim=env.action_space().n, activation=config.get("ACTIVATION", "relu"))
            pi, _ = network.apply(params, obs_batch)
            rng, _rng = jax.random.split(rng)
            actions = pi.sample(seed=_rng)
            # Convert action format
            env_act = {k: v.squeeze() for k, v in unbatchify(
                actions, env.agents, 1, env.num_agents
            ).items()}
        else:
            obs_batch = jnp.stack([obs[a] for a in env.agents])
            env_act = {}
            network = [ActorCritic(action_dim=env.action_space().n, activation=config.get("ACTIVATION", "relu")) for _ in range(env.num_agents)]
            for i in range(env.num_agents):
                obs = jnp.expand_dims(obs_batch[i], axis=0)
                pi, _ = network[i].apply(params[i], obs)
                rng, _rng = jax.random.split(rng)
                single_action = pi.sample(seed=_rng)
                env_act[env.agents[i]] = single_action

        # Execute actions
        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(_rng, state, [v.item() for v in env_act.values()])
        done = done["__all__"]

        # Render
        img = env.render(state)
        pics.append(img)

    # Save GIF
    print(f"Saving Episode GIF")
    pics = [Image.fromarray(np.array(img)) for img in pics]
    n_agents = len(env.agents)
    gif_path = f"{root_dir}/{n_agents}-agents_seed-{config['SEED']}_frames-{o_t + 1}.gif"
    pics[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        optimize=False,
        append_images=pics[1:],
        duration=200,
        loop=0,
    )

    # Log the GIF to WandB
    print("Logging GIF to WandB")
    wandb.log({"Episode GIF": wandb.Video(gif_path, caption="Evaluation Episode", format="gif")})


def evaluate_mappo_style(params, env, save_path, config, use_actor_only=True):
    """
    Evaluation function for MAPPO/IRAT/SVO/TRANSFER algorithms.

    These algorithms use simpler evaluation without PARAMETER_SHARING logic
    and without WandB GIF logging.

    Args:
        params: Model parameters
        env: Environment instance
        save_path: Path where params were saved (unused but kept for API compatibility)
        config: Configuration dictionary
        use_actor_only: If True, use Actor network; if False, use ActorCritic (for SVO/TRANSFER)
    """
    rng = jax.random.PRNGKey(0)

    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng)
    done = False

    pics = []
    img = env.render(state)
    pics.append(img)

    # Extract environment name for root_dir
    env_name = config["ENV_NAME"]
    # Map environment names to evaluation directories
    env_dir_mapping = {
        "clean_up": "cleanup",
        "coin_game": "coins",
        "coop_mining": "coop_mining",
        "gift": "gift",
        "harvest_common_open": "harvest_common",
        "harvest_common_closed": "harvest_closed",
        "harvest_common_partnership": "harvest_partnership",
        "mushrooms": "mushrooms",
        "pd_arena": "pd_arena",
        "territory_open": "territory_open",
        "harvest_open": "harvest_open",
        "harvest_closed": "harvest_closed",
        "harvest_partnership": "harvest_partnership",
    }
    env_dir = env_dir_mapping.get(env_name, env_name)
    root_dir = f"evaluation/{env_dir}"
    path = Path(root_dir + "/state_pics")
    path.mkdir(parents=True, exist_ok=True)

    for o_t in range(config["GIF_NUM_FRAMES"]):
        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *env.observation_space()[0].shape)

        # Use model to select actions
        if use_actor_only:
            # MAPPO uses SmallActor (features=16), not Actor (features=64)
            network = SmallActor(action_dim=env.action_space().n, activation=config.get("ACTIVATION", "relu"))
            pi = network.apply(params, obs_batch)
        else:
            network = ActorCritic(action_dim=env.action_space().n, activation=config.get("ACTIVATION", "relu"))
            pi, _ = network.apply(params, obs_batch)

        rng, _rng = jax.random.split(rng)
        actions = pi.sample(seed=_rng)

        # Convert action format
        env_act = {k: v.squeeze() for k, v in unbatchify(
            actions, env.agents, 1, env.num_agents
        ).items()}

        # Execute actions
        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(_rng, state, [v.item() for v in env_act.values()])
        done = done["__all__"]

        # Render
        img = env.render(state)
        pics.append(img)

    # Save GIF
    print(f"Saving Episode GIF")
    pics = [Image.fromarray(np.array(img)) for img in pics]
    pics[0].save(
        f"{root_dir}/state_outer_step_{o_t+1}.gif",
        format="GIF",
        save_all=True,
        optimize=False,
        append_images=pics[1:],
        duration=200,
        loop=0,
    )
