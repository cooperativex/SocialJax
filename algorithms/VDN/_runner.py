"""Shared VDN runner: single_run / tune glue, factored out of the 9 per-env files.

VDN differs from the PPO-style families:
  - The config nests algorithm hyperparameters under `alg.*`; single_run / tune
    flatten that namespace before use.
  - make_train has the signature make_train(config, env) — the env is built once
    here via env_from_config and passed in.
  - Sweep flag is HYP_TUNE (not TUNE); the train.py dispatcher already handles this.
  - alg_name is hard-coded to "vdn_cnn" (the on-disk algo prefix); per-env files
    therefore do not need to supply any SINGLE_RUN_KWARGS / TUNE_KWARGS.
"""
import copy
import os

import jax
from omegaconf import OmegaConf
import wandb

import socialjax
from socialjax.wrappers.baselines import LogWrapper

ALG_NAME = "vdn_cnn"


def env_from_config(config):
    """Create SocialJax environment from config."""
    env_name = config["ENV_NAME"]
    env = socialjax.make(env_name, **config["ENV_KWARGS"])
    env = LogWrapper(env, replace_info=False)
    return env, env_name


def single_run(config, make_train):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    config = {**config, **config["alg"]}  # flatten alg.* into top level

    env, env_name = env_from_config(copy.deepcopy(config))

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            ALG_NAME.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=f"{ALG_NAME}_{env_name}",
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config, env)))
    outs = jax.block_until_ready(train_vjit(rngs))

    if config.get("SAVE_PATH", None) is not None:
        from socialjax.wrappers.baselines import save_params

        model_state = outs["runner_state"][0]
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(save_dir, f'{ALG_NAME}_{env_name}_seed{config["SEED"]}_config.yaml'),
        )
        for i, rng in enumerate(rngs):
            params = jax.tree.map(lambda x: x[i], model_state.params)
            save_path = os.path.join(
                save_dir, f'{ALG_NAME}_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors',
            )
            save_params(params, save_path)


def tune(default_config, make_train):
    """Hyperparameter sweep with wandb."""
    default_config = OmegaConf.to_container(default_config)
    default_config = {**default_config, **default_config["alg"]}
    env, env_name = env_from_config(default_config)

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])
        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v
        print("running experiment with params:", config)

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config, env)))
        outs = jax.block_until_ready(train_vjit(rngs))

    sweep_config = {
        "name": f"{ALG_NAME}_{env_name}",
        "method": "bayes",
        "metric": {
            "name": "test_returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {"values": [0.005, 0.001, 0.0005, 0.0001, 0.00005]},
            "NUM_ENVS": {"values": [8, 32, 64, 128]},
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=300)
