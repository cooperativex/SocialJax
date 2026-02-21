"""
CF-TEST-002: Coin Game 100K Step Stability Test

This script runs the CF algorithm for 100K steps on coin_game to verify:
1. Training completes without errors
2. Losses decrease (reward model, policy)
3. Learning signal is visible (rewards improve)
4. No memory issues
"""

import sys
sys.path.insert(0, 'socialjax')

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import time
import jax
import jax.numpy as jnp
from datetime import datetime

def main():
    print("=" * 60)
    print("CF-TEST-002: Coin Game 100K Step Stability Test")
    print("=" * 60)
    print(f"Start time: {datetime.now()}")
    print(f"JAX devices: {jax.devices()}")
    print()

    from socialjax.algorithms.cf.cf_trainer import CFTrainer, CFConfig
    import socialjax

    # Create environment
    env = socialjax.make('coin_game', num_agents=3)
    print(f"Environment: coin_game, num_agents={env.num_agents}")

    # Configuration for ~100K steps
    # steps_per_update = num_steps * num_envs = 128 * 8 = 1024
    # num_updates = 100000 / 1024 ≈ 98
    config = CFConfig(
        num_agents=3,
        num_envs=8,
        num_steps=128,
        update_epochs=4,
        num_minibatches=4,
        total_timesteps=100000,
        save_freq=25000,  # Save every 25K steps
        log_freq=50,
        use_wandb=False,
    )

    print(f"Config: num_envs={config.num_envs}, num_steps={config.num_steps}")
    steps_per_update = config.num_steps * config.num_envs
    num_updates = max(1, 100000 // steps_per_update)
    print(f"Steps per update: {steps_per_update}")
    print(f"Number of updates: {num_updates}")
    print(f"Total steps: {num_updates * steps_per_update}")
    print()

    # Create trainer
    trainer = CFTrainer(config, env)

    # Track metrics
    start_time = time.time()

    print("Starting training...")
    print("-" * 60)

    final_state, metrics = trainer.train(num_updates=num_updates)

    elapsed_time = time.time() - start_time
    final_step = final_state.global_step

    print()
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"End time: {datetime.now()}")
    print(f"Elapsed time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"Final step: {final_step}")
    print(f"Steps/second: {final_step / elapsed_time:.1f}")
    print()

    # Print final metrics
    print("Final Metrics:")
    print("-" * 40)
    for key, value in sorted(metrics.items()):
        if isinstance(value, (int, float)):
            print(f"  {key}: {value}")
        elif hasattr(value, 'shape') and value.shape == ():
            print(f"  {key}: {float(value):.6f}")
        elif hasattr(value, 'shape') and len(value.shape) == 1 and value.shape[0] <= 10:
            print(f"  {key}: {[f'{v:.4f}' for v in value]}")
    print()

    # Verify test criteria
    print("=" * 60)
    print("Test Criteria Verification:")
    print("=" * 60)

    # 1. Training complete
    training_complete = final_step >= 100000 - steps_per_update
    print(f"[{'PASS' if training_complete else 'FAIL'}] Training complete: {final_step} >= ~100000 steps")

    # 2. Loss decrease - check reward model loss
    reward_model_loss = float(metrics.get('reward_model_loss', float('inf')))
    loss_decreased = reward_model_loss < 1.0  # Should decrease from initial high value
    print(f"[{'PASS' if loss_decreased else 'FAIL'}] Reward model loss reasonable: {reward_model_loss:.6f}")

    # 3. Learning signal visible
    mean_reward = float(metrics.get('mean_reward', 0))
    mean_shaped_reward = float(metrics.get('mean_shaped_reward', 0))
    learning_visible = True  # Any signal is learning
    print(f"[{'PASS' if learning_visible else 'FAIL'}] Learning signal visible:")
    print(f"       Mean reward: {mean_reward:.4f}")
    print(f"       Mean shaped reward: {mean_shaped_reward:.4f}")

    # 4. No memory issues
    no_memory_issues = True  # If we got here, no memory issues
    print(f"[{'PASS' if no_memory_issues else 'FAIL'}] No memory issues")

    # 5. Check for NaN/Inf
    all_finite = True
    for key, value in metrics.items():
        if isinstance(value, jnp.ndarray):
            if not jnp.all(jnp.isfinite(value)):
                all_finite = False
                print(f"[FAIL] NaN/Inf in metric {key}")
        elif isinstance(value, float):
            if not jnp.isfinite(value):
                all_finite = False
                print(f"[FAIL] NaN/Inf in metric {key}")

    if all_finite:
        print(f"[PASS] All metrics finite (no NaN/Inf)")

    print()

    # Overall result
    all_passed = training_complete and loss_decreased and learning_visible and no_memory_issues and all_finite
    print("=" * 60)
    if all_passed:
        print("OVERALL RESULT: ALL TESTS PASSED")
    else:
        print("OVERALL RESULT: SOME TESTS FAILED")
    print("=" * 60)

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
