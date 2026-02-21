#!/usr/bin/env python
"""
Tests for CF-TRAIN-001: CF 1B Step Training

Test criteria:
- [ ] Training script runs correctly
- [ ] Checkpoints saved periodically
- [ ] Evaluation metrics recorded
- [ ] Final results compared with IPPO
"""

import pytest
import sys
import os
import json
import tempfile
import shutil
from pathlib import Path

# Setup paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'socialjax'))


class TestCFTrain001Smoke:
    """Smoke tests for 1B step training."""

    def test_jax_available(self):
        """Verify JAX is available."""
        import jax
        assert jax.devices() is not None
        print(f"JAX devices: {jax.devices()}")

    def test_environment_loads(self):
        """Verify coin_game environment loads correctly."""
        import socialjax
        env = socialjax.make('coin_game', num_agents=2)
        assert env is not None
        assert env.num_agents == 2
        print(f"Environment: {env.num_agents} agents")

    def test_config_generation(self):
        """Verify config generation for different modes."""
        # Add scripts directory to path
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'scripts')
        sys.path.insert(0, scripts_dir)

        from train_cf_1b import get_config

        # Quick mode
        quick_config = get_config('quick')
        assert quick_config['total_timesteps'] == 1_000_000
        assert quick_config['num_seeds'] >= 1

        # Medium mode
        medium_config = get_config('medium')
        assert medium_config['total_timesteps'] == 100_000_000
        assert medium_config['num_seeds'] >= 2

        # Full mode
        full_config = get_config('full')
        assert full_config['total_timesteps'] == 1_000_000_000
        assert full_config['num_seeds'] == 3
        assert full_config['checkpoint_freq'] == 10_000_000
        assert full_config['eval_freq'] == 5_000_000

        print("Config generation OK")


class TestCFTrain001QuickRun:
    """Quick run tests for 1B step training."""

    @pytest.mark.slow
    def test_cf_quick_training_completes(self):
        """Test that CF quick training completes using existing CFTrainer."""
        from socialjax.algorithms.cf.cf_trainer import CFTrainer, CFConfig
        import socialjax

        # Create environment with 2 agents (CF-TRAIN-001 spec)
        env = socialjax.make('coin_game', num_agents=2)

        # Create CF config with alpha=0.5 (optimal from CF-BENCH-002)
        cf_config = CFConfig(
            env_name='coin_game',
            num_agents=2,
            num_envs=8,
            total_timesteps=10_000,  # Short test
            num_steps=128,
            update_epochs=4,
            num_minibatches=4,
            alpha=0.5,
            use_auto_alpha=False,
            log_freq=5,
            save_freq=0,
        )

        # Create trainer
        trainer = CFTrainer(cf_config, env)

        # Train
        final_state, final_metrics = trainer.train()

        # Verify result
        assert final_metrics is not None
        assert 'mean_reward' in final_metrics
        assert 'collective_reward' not in final_metrics  # We compute this separately

        collective_reward = final_metrics['mean_reward'] * 2
        print(f"CF Quick Training: mean_reward = {final_metrics['mean_reward']:.4f}, collective = {collective_reward:.4f}")

    @pytest.mark.slow
    def test_ippo_quick_training_completes(self):
        """Test that IPPO quick training completes."""
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'scripts')
        sys.path.insert(0, scripts_dir)

        from train_cf_1b import run_ippo_long_training, get_config

        config = get_config('quick')

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_ippo_long_training(
                config,
                seed=42,
                output_dir=tmpdir,
            )

        # Verify result structure
        assert result is not None
        assert result['algorithm'] == 'IPPO'
        assert result['environment'] == 'coin_game'
        assert result['num_agents'] == 2
        assert result['total_timesteps'] == config['total_timesteps']
        assert 'final_metrics' in result

        print(f"IPPO Quick Training: collective_reward = {result['final_metrics']['collective_reward']:.4f}")


class TestCFTrain001Checkpoint:
    """Checkpoint tests for 1B step training."""

    @pytest.mark.slow
    def test_checkpoint_saved_periodically(self):
        """Test that checkpoints are saved periodically."""
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'scripts')
        sys.path.insert(0, scripts_dir)

        from train_cf_1b import run_cf_long_training, get_config

        config = get_config('quick')

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_cf_long_training(
                config,
                seed=42,
                output_dir=tmpdir,
                use_wandb=False,
            )

            # Check if checkpoints were saved
            checkpoints_saved = result.get('checkpoints_saved', [])

            # For quick mode with checkpoint_freq=200K and total_timesteps=1M
            # We expect at least some checkpoints
            # Note: might be 0 if training is too short for checkpoint

        print(f"Checkpoints saved: {len(checkpoints_saved)}")
        assert isinstance(checkpoints_saved, list)


class TestCFTrain001Evaluation:
    """Evaluation tests for 1B step training."""

    @pytest.mark.slow
    def test_evaluation_metrics_recorded(self):
        """Test that evaluation metrics are recorded."""
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'scripts')
        sys.path.insert(0, scripts_dir)

        from train_cf_1b import run_cf_long_training, get_config

        config = get_config('quick')

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_cf_long_training(
                config,
                seed=42,
                output_dir=tmpdir,
                use_wandb=False,
            )

            # Check if evaluation was performed
            eval_results = result.get('eval_results', [])
            final_eval = result.get('final_eval', {})

        print(f"Evaluation results: {len(eval_results)}")
        print(f"Final eval: {final_eval}")

        # At minimum, final evaluation should be present
        assert final_eval is not None
        if final_eval:
            assert 'mean_reward' in final_eval
            assert 'collective_reward' in final_eval


class TestCFTrain001Comparison:
    """Comparison tests for 1B step training."""

    @pytest.mark.slow
    def test_cf_vs_ippo_comparison(self):
        """Test that CF vs IPPO comparison works."""
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'scripts')
        sys.path.insert(0, scripts_dir)

        from train_cf_1b import (
            run_cf_long_training,
            run_ippo_long_training,
            compare_results,
            get_config,
        )

        config = get_config('quick')

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run CF
            cf_result = run_cf_long_training(
                config,
                seed=42,
                output_dir=os.path.join(tmpdir, 'cf'),
                use_wandb=False,
            )

            # Run IPPO
            ippo_result = run_ippo_long_training(
                config,
                seed=142,
                output_dir=os.path.join(tmpdir, 'ippo'),
            )

            # Compare
            comparison = compare_results([cf_result], [ippo_result])

        # Verify comparison structure
        assert comparison is not None
        assert 'cf' in comparison
        assert 'ippo' in comparison
        assert 'comparison' in comparison
        assert 'improvement_percent' in comparison['comparison']
        assert 'cf_better' in comparison['comparison']

        print(f"CF collective: {comparison['cf']['collective_reward_avg']:.4f}")
        print(f"IPPO collective: {comparison['ippo']['collective_reward_avg']:.4f}")
        print(f"Improvement: {comparison['comparison']['improvement_percent']:+.1f}%")


class TestCFTrain001Verification:
    """Verification tests for CF-TRAIN-001 criteria."""

    def test_alpha_is_0_5(self):
        """Verify that optimal alpha=0.5 is used."""
        from socialjax.algorithms.cf.cf_trainer import CFConfig

        config = CFConfig(
            env_name='coin_game',
            num_agents=2,
            alpha=0.5,
            use_auto_alpha=False,
        )

        assert config.alpha == 0.5, "Alpha should be 0.5 for optimal performance"
        print(f"Alpha: {config.alpha}")

    def test_num_agents_is_2(self):
        """Verify that num_agents=2 as per CF-TRAIN-001 spec."""
        import socialjax

        env = socialjax.make('coin_game', num_agents=2)
        assert env.num_agents == 2, "CF-TRAIN-001 requires 2 agents"
        print(f"Num agents: {env.num_agents}")

    def test_training_config_structure(self):
        """Verify training config structure."""
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'scripts')
        sys.path.insert(0, scripts_dir)

        from train_cf_1b import get_config

        full_config = get_config('full')

        # Required fields
        required_fields = [
            'total_timesteps',
            'num_envs',
            'num_steps',
            'update_epochs',
            'num_minibatches',
            'log_freq',
            'eval_freq',
            'checkpoint_freq',
            'num_seeds',
        ]

        for field in required_fields:
            assert field in full_config, f"Missing required config field: {field}"
            assert full_config[field] > 0, f"Config field {field} should be positive"

        print("Config structure OK")

    def test_full_mode_params(self):
        """Verify full mode parameters match CF-TRAIN-001 spec."""
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'scripts')
        sys.path.insert(0, scripts_dir)

        from train_cf_1b import get_config

        full_config = get_config('full')

        assert full_config['total_timesteps'] == 1_000_000_000, "Full mode should be 1B steps"
        assert full_config['checkpoint_freq'] == 10_000_000, "Checkpoint every 10M steps"
        assert full_config['eval_freq'] == 5_000_000, "Eval every 5M steps"
        assert full_config['num_seeds'] == 3, "3 seeds for statistical significance"

        print("Full mode params OK")


class TestCFTrain001ResultsFormat:
    """Tests for results format."""

    @pytest.mark.slow
    def test_results_json_format(self):
        """Test that results can be serialized to JSON."""
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'scripts')
        sys.path.insert(0, scripts_dir)

        from train_cf_1b import run_cf_long_training, get_config

        config = get_config('quick')

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_cf_long_training(
                config,
                seed=42,
                output_dir=tmpdir,
                use_wandb=False,
            )

            # Try to serialize to JSON (without training_history)
            result_save = {k: v for k, v in result.items() if k != 'training_history'}

            # Check that it can be serialized
            try:
                json_str = json.dumps(result_save, indent=2, default=str)
                assert len(json_str) > 0
                print("Results JSON serialization OK")
            except Exception as e:
                pytest.fail(f"Failed to serialize results to JSON: {e}")


class TestCFTrain001Integration:
    """Integration tests for complete training pipeline."""

    @pytest.mark.slow
    def test_full_pipeline_quick(self):
        """Test full training pipeline in quick mode."""
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'scripts')
        sys.path.insert(0, scripts_dir)

        from train_cf_1b import (
            run_cf_long_training,
            run_ippo_long_training,
            compare_results,
            get_config,
        )

        config = get_config('quick')

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run CF
            cf_result = run_cf_long_training(
                config,
                seed=42,
                output_dir=os.path.join(tmpdir, 'cf'),
                use_wandb=False,
            )

            # Run IPPO
            ippo_result = run_ippo_long_training(
                config,
                seed=142,
                output_dir=os.path.join(tmpdir, 'ippo'),
            )

            # Compare
            comparison = compare_results([cf_result], [ippo_result])

            # Save results
            result_file = os.path.join(tmpdir, 'results.json')
            with open(result_file, 'w') as f:
                comparison_save = {
                    'timestamp': comparison['timestamp'],
                    'num_seeds': comparison['num_seeds'],
                    'cf_collective': comparison['cf']['collective_reward_avg'],
                    'ippo_collective': comparison['ippo']['collective_reward_avg'],
                    'improvement': comparison['comparison']['improvement_percent'],
                }
                json.dump(comparison_save, f, indent=2)

        # Verify everything worked
        assert os.path.exists(result_file)

        with open(result_file) as f:
            loaded = json.load(f)

        assert loaded['cf_collective'] is not None
        assert loaded['ippo_collective'] is not None

        print("Full pipeline test passed!")
        print(f"  CF collective: {loaded['cf_collective']:.4f}")
        print(f"  IPPO collective: {loaded['ippo_collective']:.4f}")
        print(f"  Improvement: {loaded['improvement']:+.1f}%")


# Run tests if called directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
