from socialjax.environments import (
    # Social dilemma environments
    Territory_open,
    Harvest_open,
    Clean_up,
    CoopMining,
    CoinGame,
    Mushrooms,
    Gift,
    PD_Arena,
)

# Registry of all available environments
REGISTERED_ENVS = [
    # Social dilemma environments
    "coin_game",
    "harvest_common_open",
    # "harvest_common_closed",
    "clean_up",
    "coop_mining",
    "territory_open",
    "pd_arena",
    "mushrooms",
    "gift",
]


def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's env.make(env_name), built off Gymnax"""
    if env_id not in REGISTERED_ENVS:
        raise ValueError(f"{env_id} is not in registered SocialJax environments")

    elif env_id == "harvest_common_open":
        env = Harvest_open(**env_kwargs)
    # elif env_id == "harvest_common_closed":
    #     env = Harvest_closed(**env_kwargs)
    elif env_id == "clean_up":
        env = Clean_up(**env_kwargs)
    elif env_id == "coop_mining":
        env = CoopMining(**env_kwargs)
    elif env_id == "territory_open":
        env = Territory_open(**env_kwargs)
    elif env_id == "pd_arena":
        env = PD_Arena(**env_kwargs)
    elif env_id == "coin_game":
        env = CoinGame(**env_kwargs)
    elif env_id == "mushrooms":
        env = Mushrooms(**env_kwargs)
    elif env_id == "gift":
        env = Gift(**env_kwargs)
    return env