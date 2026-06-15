"""Shared IRAT runner: single_run / tune glue, factored out of the 9 per-env files.

IRAT specifics:
  - 4 networks (ind_actor, ind_critic, team_actor, team_critic). The runner saves
    the team_actor (index 2) for evaluation: out["runner_state"][0][0][2].
  - tags=["IRAT", "INDIVIDUAL_REWARD"]; no wandb group.
  - filename has _reward_<REWARD> suffix.
  - No jax.jit wrapping on make_train (per the originals).
  - Evaluation uses evaluate_mappo_style.
"""
import copy

import jax
from omegaconf import OmegaConf
import wandb

import socialjax
from algorithms.utils import save_params, load_params, evaluate_mappo_style as evaluate


def single_run(config, make_train, *, wandb_name):
    config = OmegaConf.to_container(config)
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IRAT", "INDIVIDUAL_REWARD"],
        config=config,
        mode=config["WANDB_MODE"],
        name=wandb_name,
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = make_train(config)
    out = jax.vmap(train_jit)(rngs)

    print("** Saving Results **")
    filename = f'{config["ENV_NAME"]}_seed{config["SEED"]}_reward_{config["REWARD"]}'
    # IRAT: save team_actor (index 2) for evaluation
    team_actor_train_state = jax.tree.map(lambda x: x[0], out["runner_state"][0][0][2])
    save_path = f"./checkpoints/{filename}.pkl"
    save_params(team_actor_train_state, save_path)
    params = load_params(save_path)
    print("** Evaluating Results **")
    evaluate(params, socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]), save_path, config)


def tune(default_config, make_train, *, sweep_name):
    default_config = OmegaConf.to_container(default_config)

    sweep_config = {
        "name": sweep_name,
        "method": "grid",
        "metric": {
            "name": "returned_episode_original_returns",
            "goal": "maximize",
        },
        "parameters": {
            "SEED": {"values": [42, 52, 62]},
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
