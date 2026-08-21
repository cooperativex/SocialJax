"""Generate 500-step evaluation GIFs from the trained IPPO checkpoints.

Usage:
  JAX_PLATFORMS=cpu python generate_gifs.py [env ...]   # default: all
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

REPO = Path("/lus/lfs1aip2/projects/a5l/zihao/SocialJax")
sys.path.insert(0, str(REPO))

import socialjax
from algorithms.utils.networks import ActorCritic
from algorithms.utils.io_utils import load_params
from algorithms.utils.data_utils import unbatchify

STEPS = 500
OUT_DIR = REPO / "evaluation" / "gifs_500steps"
CKPT_DIR = REPO / "checkpoints" / "indvidual"

# config name -> checkpoint env prefixes to try, in order (post-ENV_LABEL-fix
# name first, legacy name as fallback)
ENVS = {
    "coins": ["coin_game"],
    "cleanup": ["clean_up"],
    "coop_mining": ["coop_mining"],
    "gift": ["gift"],
    "mushrooms": ["mushrooms"],
    "pd_arena": ["pd_arena"],
    "harvest_open": ["harvest_open"],
    "harvest_closed": ["harvest_closed"],
    # legacy harvest_common_open file holds partnership weights (it finished
    # last before the ENV_LABEL fix)
    "harvest_partnership": ["harvest_partnership", "harvest_common_open"],
}


def rollout_gif(cfg_name, ckpt_prefixes, mode):
    with initialize_config_dir(config_dir=str(REPO / "algorithms/IPPO/config"), version_base=None):
        cfg = compose(config_name=f"ippo_cnn_{cfg_name}", overrides=[f"reward={mode}"])
    cfg = OmegaConf.to_container(cfg, resolve=True)

    ckpt = next((p for c in ckpt_prefixes
                 if (p := CKPT_DIR / f"{c}_seed30_reward_{mode}.pkl").exists()), None)
    if ckpt is None:
        print(f"SKIP {cfg_name} {mode}: no checkpoint for {ckpt_prefixes}")
        return
    params = load_params(str(ckpt))

    env = socialjax.make(cfg["ENV_NAME"], **cfg["ENV_KWARGS"])
    network = ActorCritic(action_dim=env.action_space().n, activation=cfg.get("ACTIVATION", "relu"))

    rng = jax.random.PRNGKey(0)
    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng)

    pics = [np.array(env.render(state))]
    for t in range(STEPS):
        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *env.observation_space()[0].shape)
        pi, _ = network.apply(params, obs_batch)
        rng, _rng = jax.random.split(rng)
        actions = pi.sample(seed=_rng)
        env_act = {k: v.squeeze() for k, v in unbatchify(actions, env.agents, 1, env.num_agents).items()}
        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(_rng, state, [v.item() for v in env_act.values()])
        pics.append(np.array(env.render(state)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = [Image.fromarray(p) for p in pics]
    out = OUT_DIR / f"{cfg_name}_{mode}_{STEPS}steps.gif"
    frames[0].save(out, format="GIF", save_all=True, optimize=False,
                   append_images=frames[1:], duration=100, loop=0)
    print(f"OK {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    # harvest_open/harvest_closed checkpoints were overwritten by
    # harvest_partnership (same save filename) — regenerate those after retraining.
    valid = [e for e in ENVS if e not in ("harvest_open", "harvest_closed")]
    targets = sys.argv[1:] or valid
    for name in targets:
        for mode in ("common", "individual"):
            try:
                rollout_gif(name, ENVS[name], mode)
            except Exception as e:
                print(f"FAIL {name} {mode}: {type(e).__name__}: {e}")
