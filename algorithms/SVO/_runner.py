"""Shared SVO runner: single_run / tune glue, factored out of the 9 per-env files.

Each algorithms/SVO/svo_cnn_<env>.py defines its own `make_train(config)` and
delegates to single_run()/tune() here:

    @hydra.main(version_base=None, config_path="config", config_name="svo_cnn_coin")
    def main(config):
        if config["TUNE"]:
            tune(config, make_train, sweep_name="coin_angle")
        else:
            single_run(config, make_train, wandb_name="svo_cnn_coin", group_name="coin")
"""
import copy

import jax
from omegaconf import OmegaConf
import wandb

import socialjax
from algorithms.utils import save_params, load_params, evaluate_ippo as evaluate


def single_run(config, make_train, *, wandb_name, group_name):
    """One training run, saving + evaluating at the end."""
    config = OmegaConf.to_container(config)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["SVO", "FF"],
        config=config,
        mode=config["WANDB_MODE"],
        name=wandb_name,
        group=group_name,
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)

    print("** Saving Results **")
    filename = f'{config["ENV_NAME"]}_seed{config["SEED"]}_reward_{config["REWARD"]}'
    train_state = jax.tree.map(lambda x: x[0], out["runner_state"][0])
    save_path = f"./checkpoints/{filename}.pkl"
    save_params(train_state, save_path)
    params = load_params(save_path)

    evaluate(params, socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]), save_path, config)


def tune(default_config, make_train, *, sweep_name):
    """Hyperparameter sweep with wandb."""
    default_config = OmegaConf.to_container(default_config)

    sweep_config = {
        "name": sweep_name,
        "method": "grid",
        "metric": {
            "name": "returned_episode_original_returns",
            "goal": "maximize",
        },
        "parameters": {
            # "LR": {"values": [0.001, 0.0005, 0.0001, 0.00005]},
            # "ACTIVATION": {"values": ["relu", "tanh"]},
            # "UPDATE_EPOCHS": {"values": [2, 4, 8]},
            # "NUM_MINIBATCHES": {"values": [4, 8, 16, 32]},
            # "CLIP_EPS": {"values": [0.1, 0.2, 0.3]},
            # "ENT_COEF": {"values": [0.001, 0.01, 0.1]},
            # "NUM_STEPS": {"values": [64, 128, 256]},
            "ENV_KWARGS.svo_w": {"values": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]},
            # "ENV_KWARGS.svo_ideal_angle_degrees": {"values": [0, 45, 90]},
        },
    }

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])
        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            if "." in k:
                parent, child = k.split(".", 1)
                config[parent][child] = v
            else:
                config[k] = v

        run_name = f"sweep_{config['ENV_NAME']}_seed{config['SEED']}"
        wandb.run.name = run_name
        print("Running experiment:", run_name)

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        outs = jax.block_until_ready(train_vjit(rngs))
        train_state = jax.tree.map(lambda x: x[0], outs["runner_state"][0])

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)
