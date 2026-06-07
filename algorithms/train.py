"""Unified CLI entry point for all algorithms.

Usage:
    python algorithms/train.py --algo IPPO --env coins
    python algorithms/train.py --algo IPPO --env coins SEED=42 NUM_ENVS=128
    python algorithms/train.py --algo SVO  --env harvest_open
    python algorithms/train.py --algo VDN  --env cleanup alg.NUM_ENVS=32

Per-env modules (e.g. algorithms/IPPO/ippo_cnn_coins.py) expose:
    - make_train(config)              the family training loop, unchanged from
                                      the original per-env implementation
    - SINGLE_RUN_KWARGS = {...}        kwargs forwarded to <family>._runner.single_run
    - TUNE_KWARGS = {...}              kwargs forwarded to <family>._runner.tune

Per-family <family>/_runner.py exports single_run(config, make_train, **kwargs)
and tune(config, make_train, **kwargs).
"""
import argparse
import importlib
import sys

# (canonical algo prefix used in per-env module / config filename)
ALGO_PREFIX = {
    "IPPO":     "ippo",
    "SVO":      "svo",
    "MAPPO":    "mappo",
    "IRAT":     "IRAT",       # preserves the original capitalization on disk
    "TRANSFER": "transfer",
    "VDN":      "vdn",
}


def _parse_cli():
    """Strip --algo / --env from sys.argv; forward the rest to Hydra."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--algo", required=True, choices=list(ALGO_PREFIX))
    ap.add_argument("--env",  required=True,
                    help="env stem, e.g. coins / coin / harvest_open / pd_arena")
    ap.add_argument("-h", "--help", action="store_true")
    args, leftover = ap.parse_known_args()
    if args.help:
        ap.print_help()
        sys.exit(0)
    # Hydra reads sys.argv directly; only let it see the overrides.
    sys.argv = [sys.argv[0]] + leftover
    return args.algo, args.env


def _resolve(algo: str, env: str):
    """Return (env_module, runner_module, config_path, config_name)."""
    prefix = ALGO_PREFIX[algo]
    stem = f"{prefix}_cnn_{env}"
    env_module = f"algorithms.{algo}.{stem}"
    runner_module = f"algorithms.{algo}._runner"
    config_path = f"{algo}/config"      # relative to this file
    config_name = stem
    return env_module, runner_module, config_path, config_name


def _build_main(algo, env, env_mod, runner, config_path, config_name):
    """Create a @hydra.main-decorated entry function bound to the chosen algo/env."""
    import hydra

    tune_flag = "HYP_TUNE" if algo == "VDN" else "TUNE"
    single_kwargs = getattr(env_mod, "SINGLE_RUN_KWARGS", {})
    tune_kwargs   = getattr(env_mod, "TUNE_KWARGS",       {})

    @hydra.main(version_base=None, config_path=config_path, config_name=config_name)
    def main(cfg):
        if cfg.get(tune_flag, False):
            runner.tune(cfg, env_mod.make_train, **tune_kwargs)
        else:
            runner.single_run(cfg, env_mod.make_train, **single_kwargs)

    return main


def main():
    algo, env = _parse_cli()
    env_module, runner_module, config_path, config_name = _resolve(algo, env)

    try:
        env_mod = importlib.import_module(env_module)
    except ModuleNotFoundError as e:
        sys.exit(f"Cannot import per-env module '{env_module}': {e}")
    try:
        runner = importlib.import_module(runner_module)
    except ModuleNotFoundError as e:
        sys.exit(f"Cannot import runner '{runner_module}' (this family may not be refactored yet): {e}")

    hydra_main = _build_main(algo, env, env_mod, runner, config_path, config_name)
    hydra_main()


if __name__ == "__main__":
    main()
