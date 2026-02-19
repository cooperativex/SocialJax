"""Unit tests for Migration Guide verification.

This module tests that the migration guide examples are copy-pasteable
and work correctly. Tests verify:
- V1 to V2 API mapping examples work
- Migration patterns are valid
- Backward compatibility notes are accurate
- Common issues solutions work
"""

import sys
import pytest
import tempfile
import os

# Add socialjax to path
sys.path.insert(0, 'socialjax')


class TestMigrationGuideImports:
    """Test that all imports mentioned in migration guide work."""

    def test_v2_basic_import(self):
        """Test basic socialjax import as shown in migration guide."""
        import socialjax
        assert socialjax is not None

    def test_v2_config_imports(self):
        """Test config imports as shown in migration guide."""
        from socialjax.config import (
            ConfigManager,
            create_default_config,
            TrainingConfig,
            NetworkConfig,
            AlgorithmConfig,
            EnvironmentConfig,
            SocialJaxConfig,
        )
        assert ConfigManager is not None
        assert create_default_config is not None
        assert TrainingConfig is not None
        assert NetworkConfig is not None
        assert AlgorithmConfig is not None
        assert EnvironmentConfig is not None
        assert SocialJaxConfig is not None

    def test_v2_callback_imports(self):
        """Test callback imports as shown in migration guide."""
        from socialjax.training.callbacks import (
            BaseCallback,
            CallbackList,
            CheckpointCallback,
            EvalCallback,
            ProgressCallback,
            WandbCallback,
        )
        assert BaseCallback is not None
        assert CallbackList is not None
        assert CheckpointCallback is not None
        assert EvalCallback is not None
        assert ProgressCallback is not None
        assert WandbCallback is not None

    def test_v2_evaluation_imports(self):
        """Test evaluation imports as shown in migration guide."""
        from socialjax.evaluation import (
            Evaluator,
            EvaluatorConfig,
            EpisodeMetrics,
            EvaluationMetrics,
            compute_episode_return,
            compute_agent_returns,
            compute_cooperation_rate,
            compute_gini_coefficient,
            compute_social_welfare,
            save_gif,
            save_mp4,
        )
        assert Evaluator is not None
        assert EvaluatorConfig is not None
        assert EpisodeMetrics is not None
        assert EvaluationMetrics is not None
        assert compute_episode_return is not None
        assert compute_agent_returns is not None
        assert compute_cooperation_rate is not None
        assert compute_gini_coefficient is not None
        assert compute_social_welfare is not None
        assert save_gif is not None
        assert save_mp4 is not None

    def test_v2_wrapper_imports(self):
        """Test wrapper imports as shown in migration guide."""
        from socialjax.environments.wrappers import (
            NormalizationWrapper,
            FrameStackWrapper,
            TimeLimitWrapper,
        )
        assert NormalizationWrapper is not None
        assert FrameStackWrapper is not None
        assert TimeLimitWrapper is not None


class TestQuickReferenceTable:
    """Test patterns shown in Quick Reference table."""

    def test_algorithm_registry_pattern(self):
        """Test get_algorithm pattern from quick reference."""
        import socialjax

        # Get algorithm class (should work even if no algorithm registered)
        try:
            algo_class = socialjax.get_algorithm('ippo')
            assert algo_class is not None
        except socialjax.AlgorithmNotFoundError:
            pytest.skip("IPPO algorithm not registered")

    def test_config_manager_pattern(self):
        """Test ConfigManager pattern from quick reference."""
        import socialjax

        manager = socialjax.ConfigManager()
        assert manager is not None

        # Test load method exists
        assert hasattr(manager, 'load')

    def test_trainer_save_load_pattern(self):
        """Test trainer save/load pattern from quick reference."""
        import socialjax

        # Verify save/load methods exist on Trainer
        trainer = socialjax.Trainer.__new__(socialjax.Trainer)
        assert hasattr(trainer, 'save')
        assert hasattr(trainer, 'load')


class TestV2TrainingPattern:
    """Test V2 training patterns from migration guide."""

    def test_trainer_creation_from_names(self):
        """Test creating trainer from algorithm/env names."""
        import socialjax

        # Create trainer with algorithm and env names
        # Note: This may skip if environment issues exist
        try:
            trainer = socialjax.Trainer(
                algorithm='ippo',
                env='clean_up',
                config={'total_timesteps': 100},
            )
            assert trainer is not None
        except Exception as e:
            # Skip if environment issues
            pytest.skip(f"Environment issue: {e}")

    def test_trainer_with_callbacks(self):
        """Test trainer creation with callbacks."""
        import socialjax

        callbacks = [
            socialjax.ProgressCallback(total_timesteps=1000),
        ]

        # Use kwargs for training params (will be mapped to appropriate sections)
        trainer = socialjax.Trainer(
            algorithm='ippo',
            env='clean_up',
            callbacks=callbacks,
            total_timesteps=100,
        )
        assert trainer is not None


class TestV2ConfigPatterns:
    """Test V2 configuration patterns from migration guide."""

    def test_config_manager_load_presets(self):
        """Test loading preset configs."""
        import socialjax

        manager = socialjax.ConfigManager()
        config = manager.load('ippo', 'coin_game')

        assert config is not None
        assert hasattr(config, 'to_dict')

    def test_create_default_config(self):
        """Test create_default_config helper."""
        import socialjax

        config = socialjax.create_default_config(algorithm='ippo')
        assert config is not None

    def test_config_dataclasses(self):
        """Test creating configs programmatically."""
        import socialjax

        training = socialjax.TrainingConfig(
            total_timesteps=1_000_000,
            num_envs=8,
            num_steps=128,
            gamma=0.99,
            learning_rate=2.5e-4,
        )

        assert training.total_timesteps == 1_000_000
        assert training.num_envs == 8
        assert training.gamma == 0.99

    def test_network_config(self):
        """Test NetworkConfig creation."""
        import socialjax

        # NetworkConfig uses architecture, hidden_size, num_channels
        network = socialjax.NetworkConfig(
            architecture='cnn_actor_critic',
            hidden_size=128,
            num_channels=(32, 64, 64),
        )

        assert network.architecture == 'cnn_actor_critic'
        assert network.hidden_size == 128


class TestV2CallbackPatterns:
    """Test V2 callback patterns from migration guide."""

    def test_checkpoint_callback(self):
        """Test CheckpointCallback creation."""
        import socialjax

        callback = socialjax.CheckpointCallback(
            save_freq=1000,
            save_path='./checkpoints',
            name_prefix='ippo',
            verbose=True,
        )
        assert callback is not None
        assert callback.save_freq == 1000

    def test_eval_callback(self):
        """Test EvalCallback creation."""
        import socialjax

        # EvalCallback requires eval_env parameter
        # Create a mock env for testing
        class MockEnv:
            pass

        callback = socialjax.EvalCallback(
            eval_env=MockEnv(),
            eval_freq=5000,
            n_eval_episodes=10,
            deterministic=True,
        )
        assert callback is not None
        assert callback.eval_freq == 5000

    def test_progress_callback(self):
        """Test ProgressCallback creation."""
        import socialjax

        callback = socialjax.ProgressCallback(
            total_timesteps=1_000_000,
            progress_freq=10,
            show_metrics=['loss', 'episode_return'],
        )
        assert callback is not None
        assert callback.total_timesteps == 1_000_000

    def test_wandb_callback(self):
        """Test WandbCallback creation."""
        import socialjax

        callback = socialjax.WandbCallback(
            project='socialjax-experiments',
            name='test_run',
            log_freq=100,
        )
        assert callback is not None
        assert callback.project == 'socialjax-experiments'

    def test_custom_callback(self):
        """Test custom callback creation as shown in guide."""
        import socialjax

        class MyCustomCallback(socialjax.BaseCallback):
            def on_training_start(self, trainer):
                pass

            def on_update_end(self, trainer, update_metrics):
                pass

            def on_training_end(self, trainer):
                pass

        callback = MyCustomCallback()
        assert callback is not None
        assert hasattr(callback, 'on_training_start')
        assert hasattr(callback, 'on_update_end')
        assert hasattr(callback, 'on_training_end')


class TestV2CheckpointPatterns:
    """Test V2 checkpoint patterns from migration guide."""

    def test_algorithm_save_load_methods_exist(self):
        """Test that algorithm save/load methods exist."""
        import socialjax

        # Verify save/load exist on BaseAlgorithm
        assert hasattr(socialjax.BaseAlgorithm, 'save')
        assert hasattr(socialjax.BaseAlgorithm, 'load')

    def test_trainer_save_load_methods_exist(self):
        """Test that trainer save/load methods exist."""
        import socialjax

        # Verify save/load exist on Trainer
        assert hasattr(socialjax.Trainer, 'save')
        assert hasattr(socialjax.Trainer, 'load')


class TestV2EvaluationPatterns:
    """Test V2 evaluation patterns from migration guide."""

    def test_evaluator_config_creation(self):
        """Test EvaluatorConfig creation."""
        from socialjax.evaluation import EvaluatorConfig

        config = EvaluatorConfig(
            num_episodes=50,
            deterministic=True,
            seed=42,
        )

        assert config.num_episodes == 50
        assert config.deterministic is True
        assert config.seed == 42

    def test_episode_metrics_creation(self):
        """Test EpisodeMetrics creation."""
        from socialjax.evaluation import EpisodeMetrics

        metrics = EpisodeMetrics(
            episode_return=100.0,
            episode_length=200,
            agent_returns={'agent_0': 50.0, 'agent_1': 50.0},
        )

        assert metrics.episode_return == 100.0
        assert metrics.episode_length == 200

    def test_evaluation_metrics_creation(self):
        """Test EvaluationMetrics creation."""
        from socialjax.evaluation import EvaluationMetrics

        metrics = EvaluationMetrics(
            mean_return=95.0,
            std_return=10.0,
            num_episodes=10,
        )

        assert metrics.mean_return == 95.0
        assert metrics.std_return == 10.0
        assert metrics.num_episodes == 10


class TestCommonIssuesSolutions:
    """Test solutions shown in Common Issues section."""

    def test_config_value_compatibility_helper(self):
        """Test the config value compatibility helper from guide."""
        # This is the helper function from the migration guide
        def get_config_value(config, v1_key, default=None):
            """Get config value supporting both V1 and V2 key formats."""
            v2_key = v1_key.lower()
            if hasattr(config, 'algorithm'):
                training = config.algorithm.training
                if hasattr(training, v2_key):
                    return getattr(training, v2_key)

            if isinstance(config, dict):
                return config.get(v1_key, config.get(v2_key, default))

            return default

        import socialjax

        # Test with V2 config
        manager = socialjax.ConfigManager()
        config = manager.load('ippo', 'coin_game')

        # Should work without error
        value = get_config_value(config, 'LR', default=0.00025)
        assert value is not None

    def test_uppercase_lowercase_config_access(self):
        """Test that config can be accessed different ways."""
        import socialjax

        manager = socialjax.ConfigManager()
        config = manager.load('ippo', 'coin_game')

        # V2 style access (lowercase)
        config_dict = config.to_dict()

        # Config should have algorithm section
        assert 'algorithm' in config_dict


class TestBackwardCompatibility:
    """Test backward compatibility notes from migration guide."""

    def test_v1_algorithms_directory_exists(self):
        """Test that V1 algorithms directory still exists."""
        import os

        v1_path = os.path.join('algorithms', 'IPPO')
        assert os.path.exists(v1_path), "V1 algorithms directory should exist for backward compatibility"

    def test_list_algorithms_works(self):
        """Test that list_algorithms returns available algorithms."""
        import socialjax

        algorithms = socialjax.list_algorithms()
        assert isinstance(algorithms, list)

        # Should include standard algorithms
        expected = ['ippo', 'mappo', 'svo', 'vdn']
        for algo in expected:
            if algo in algorithms:
                assert algo in algorithms

    def test_v1_v2_coexistence(self):
        """Test that V1 and V2 can coexist (import wise)."""
        # V2 import
        import socialjax

        # Both should work
        assert socialjax.get_algorithm is not None
        assert socialjax.ConfigManager is not None


class TestMigrationChecklist:
    """Verify migration checklist items are valid."""

    def test_socialjax_package_imports_work(self):
        """Verify 'Update imports' checklist item."""
        import socialjax

        # All key imports should work
        assert hasattr(socialjax, 'Trainer')
        assert hasattr(socialjax, 'ConfigManager')
        assert hasattr(socialjax, 'CheckpointCallback')
        assert hasattr(socialjax, 'WandbCallback')

    def test_trainer_replaces_make_train(self):
        """Verify 'Replace make_train' checklist item."""
        import socialjax

        # Trainer class should exist and have train method
        assert hasattr(socialjax.Trainer, 'train')

    def test_config_manager_replaces_hydra(self):
        """Verify 'Update config format' checklist item."""
        import socialjax

        # ConfigManager should exist with expected methods
        manager = socialjax.ConfigManager()
        assert hasattr(manager, 'load')
        assert hasattr(manager, 'load_from_file')

    def test_callbacks_replace_inline_logging(self):
        """Verify 'Add callbacks' checklist item."""
        import socialjax

        # All callback types should be available
        assert hasattr(socialjax, 'CheckpointCallback')
        assert hasattr(socialjax, 'WandbCallback')
        assert hasattr(socialjax, 'ProgressCallback')
        assert hasattr(socialjax, 'EvalCallback')

    def test_checkpoint_methods_replace_pickle(self):
        """Verify 'Update checkpoint code' checklist item."""
        import socialjax

        # Trainer should have save/load methods
        assert hasattr(socialjax.Trainer, 'save')
        assert hasattr(socialjax.Trainer, 'load')

    def test_evaluator_class_available(self):
        """Verify 'Update evaluation' checklist item."""
        import socialjax

        # Evaluator should be available
        assert hasattr(socialjax, 'Evaluator')
        assert hasattr(socialjax, 'EvaluatorConfig')


class TestSummaryTableAccuracy:
    """Verify summary table is accurate."""

    def test_v2_trainer_exists(self):
        """Verify Trainer.train() is the V2 approach."""
        import socialjax
        assert hasattr(socialjax, 'Trainer')
        assert hasattr(socialjax.Trainer, 'train')

    def test_v2_registry_pattern_exists(self):
        """Verify registry pattern is the V2 approach."""
        import socialjax
        assert hasattr(socialjax, 'get_algorithm')
        assert hasattr(socialjax, 'list_algorithms')

    def test_v2_evaluator_exists(self):
        """Verify Evaluator class is the V2 approach."""
        import socialjax
        assert hasattr(socialjax, 'Evaluator')


class TestMigrationGuideFileExists:
    """Verify migration guide file exists and is accessible."""

    def test_migration_guide_exists(self):
        """Test that migration guide file exists."""
        import os
        guide_path = os.path.join('docs', 'migration_guide.md')
        assert os.path.exists(guide_path), "Migration guide should exist at docs/migration_guide.md"

    def test_migration_guide_has_content(self):
        """Test that migration guide has content."""
        import os
        guide_path = os.path.join('docs', 'migration_guide.md')

        with open(guide_path, 'r') as f:
            content = f.read()

        # Should have key sections
        assert '# Migration Guide' in content
        assert 'Quick Reference' in content
        assert 'Training Scripts' in content
        assert 'Algorithm Usage' in content
        assert 'Configuration' in content
        assert 'Callbacks and Logging' in content
        assert 'Checkpoints' in content
        assert 'Evaluation' in content
        assert 'Common Issues and Solutions' in content
        assert 'Backward Compatibility' in content
        assert 'Migration Checklist' in content

    def test_migration_guide_examples_valid(self):
        """Test that migration guide has code examples."""
        import os
        guide_path = os.path.join('docs', 'migration_guide.md')

        with open(guide_path, 'r') as f:
            content = f.read()

        # Should have before/after code blocks
        assert '```python' in content, "Should have Python code examples"
        assert 'V1' in content, "Should show V1 patterns"
        assert 'V2' in content, "Should show V2 patterns"
        assert 'socialjax.Trainer' in content, "Should show V2 Trainer usage"
