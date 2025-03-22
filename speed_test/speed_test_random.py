""" 
Based on jaxmarl Implementation of speed test
"""

import time

import numpy as np
import jax
import jax.numpy as jnp

import socialjax

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    """
    reshape actions for agents
    """
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def make_benchmark(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]

    def benchmark(rng):
        def init_runner_state(rng):

            # INIT ENV
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state = jax.vmap(env.reset)(reset_rng)

            return (env_state, obsv, rng)

        def env_step(runner_state, _unused):
            env_state, _, rng = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            rngs = jax.random.split(_rng, config["NUM_ACTORS"]).reshape(
                (env.num_agents, config["NUM_ENVS"], -1))
            actions = [jax.vmap(env.action_space(k).sample)(
                rngs[i]) for i, k in enumerate(env.agents)]
            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state, _, _, info = jax.vmap(env.step)(
                rng_step, env_state, actions
            )
            runner_state = (env_state, obsv, rng)
            return runner_state, None

        rng, init_rng = jax.random.split(rng)
        runner_state = init_runner_state(init_rng)
        runner_state = jax.lax.scan(env_step, runner_state, None, config["NUM_STEPS"])
        return runner_state

    return benchmark


ENV = "coin_game"

config = {
    "NUM_STEPS": 1000,
    "num_agents" : 2,
    "NUM_ENVS": 1000,
    "ACTIVATION": "relu",
    "ENV_KWARGS": {},
    "ENV_NAME": ENV,
    "NUM_SEEDS": 1,
    "SEED": 0,
}

# num_envs = [1, 128, 1024, 4096]
num_envs = [1]
sps_list = []
for num in num_envs:
    config["NUM_ENVS"] = num
    benchmark_fn = jax.jit(make_benchmark(config))
    rng = jax.random.PRNGKey(config["SEED"])
    rng, _rng = jax.random.split(rng)
    benchmark_jit = jax.jit(benchmark_fn).lower(_rng).compile()
    before = time.perf_counter_ns()
    runner_state = jax.block_until_ready(benchmark_jit(_rng))
    after = time.perf_counter_ns()
    total_time = (after - before) / 1e9

    sps = config['NUM_STEPS'] * config['NUM_ENVS'] / total_time
    sps_list.append(sps)
    print(f"socialjax, Num Envs: {num}, Total Time (s): {total_time}")
    print(f"socialjax, Num Envs: {num}, Total Steps: {config['NUM_STEPS'] * config['NUM_ENVS']}")
    print(f"socialjax, Num Envs: {num}, SPS: {sps}")
