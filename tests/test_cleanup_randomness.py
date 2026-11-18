"""Clean Up's random draws must not share keys.

Clean Up's step drew apple regrowth, the choice of dirt cell and the dirt spawn chance
from the same key, so they shared random numbers: the spawn draw always equalled apple
cell 0's regrowth draw, and the dirt-cell noise matched the apple draws too. Its reset
reused one subkey for both spawn shuffles and the starting directions.

Run with pytest, or standalone:
  export PYTHONPATH=$PWD:$PYTHONPATH
  JAX_PLATFORMS=cpu python tests/test_cleanup_randomness.py
"""

import jax
import numpy as onp

from socialjax.environments.cleanup.clean_up import Clean_up

DRAWS = ("uniform", "permutation", "randint")


def run(name, fn):
    fn()
    print(f"ok: {name}")


def record_keys(fn):
    """Run fn without jit and return (draw name, key bytes) for every random draw."""
    seen = []
    originals = {name: getattr(jax.random, name) for name in DRAWS}

    def recorder(name, orig):
        def wrapped(key, *args, **kwargs):
            if not isinstance(key, jax.core.Tracer):
                seen.append((name, onp.asarray(key).tobytes()))
            return orig(key, *args, **kwargs)
        return wrapped

    for name, orig in originals.items():
        setattr(jax.random, name, recorder(name, orig))
    try:
        with jax.disable_jit():
            fn()
    finally:
        for name, orig in originals.items():
            setattr(jax.random, name, orig)
    return seen


def test_clean_up_step_draws_use_distinct_keys():
    env = Clean_up(jit=False)
    _, state = env.reset(jax.random.PRNGKey(0))
    actions = [0] * env.num_agents
    seen = record_keys(lambda: env.step(jax.random.PRNGKey(1), state, actions))
    keys = [k for _, k in seen]
    assert len(keys) >= 3, f"expected the step's random draws, saw {len(keys)}"
    shared = len(keys) - len(set(keys))
    assert shared == 0, f"{shared} draws reused a key ({[n for n, _ in seen]})"


def test_clean_up_reset_draws_use_distinct_keys():
    env = Clean_up(jit=False)
    seen = record_keys(lambda: env.reset(jax.random.PRNGKey(0)))
    keys = [k for _, k in seen]
    assert len(keys) == 3, f"expected 2 shuffles and 1 heading draw, saw {len(keys)}"
    assert len(set(keys)) == len(keys), "reset draws reused a key"


def test_clean_up_cf_fails_clearly():
    try:
        Clean_up(cf=True)
    except NotImplementedError:
        return
    raise AssertionError("Clean_up(cf=True) should raise NotImplementedError")


if __name__ == "__main__":
    run("clean_up step draws use distinct keys", test_clean_up_step_draws_use_distinct_keys)
    run("clean_up reset draws use distinct keys", test_clean_up_reset_draws_use_distinct_keys)
    run("clean_up cf=True fails clearly", test_clean_up_cf_fails_clearly)
    print("ALL RANDOMNESS TESTS PASSED")
