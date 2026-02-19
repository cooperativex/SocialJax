"""Unit tests for API documentation verification.

This module tests that the API documentation examples are copy-pasteable
and work correctly. Tests verify:
- All public classes have docstrings
- All documented imports work
- Example code snippets run without errors
- Public API is accessible as documented
"""

import sys
import pytest

# Add socialjax to path
sys.path.insert(0, 'socialjax')


class TestPublicAPIImports:
    """Test that all public API components can be imported."""

    def test_import_socialjax(self):
        """Test basic socialjax import works."""
        import socialjax
        assert socialjax is not None

    def test_version_available(self):
        """Test that version is available."""
        import socialjax
        assert hasattr(socialjax, '__version__')
        assert socialjax.__version__ == '2.0.0'

    def test_environment_creation_exports(self):
        """Test environment creation exports."""
        import socialjax
        assert hasattr(socialjax, 'make')
        assert hasattr(socialjax, 'registered_envs')

    def test_core_exports(self):
        """Test core component exports."""
        import socialjax
        assert hasattr(socialjax, 'BaseAlgorithm')
        assert hasattr(socialjax, 'BaseTrainer')
        assert hasattr(socialjax, 'AlgorithmState')
        assert hasattr(socialjax, 'TrainerState')
        assert hasattr(socialjax, 'TrainingMetrics')
        assert hasattr(socialjax, 'jit_method')
        assert hasattr(socialjax, 'Callback')

    def test_algorithm_registry_exports(self):
        """Test algorithm registry exports."""
        import socialjax
        assert hasattr(socialjax, 'register_algorithm')
        assert hasattr(socialjax, 'get_algorithm')
        assert hasattr(socialjax, 'list_algorithms')
        assert hasattr(socialjax, 'unregister_algorithm')
        assert hasattr(socialjax, 'is_algorithm_registered')
        assert hasattr(socialjax, 'clear_algorithm_registry')
        assert hasattr(socialjax, 'AlgorithmAlreadyRegisteredError')
        assert hasattr(socialjax, 'AlgorithmNotFoundError')

    def test_network_registry_exports(self):
        """Test network registry exports."""
        import socialjax
        assert hasattr(socialjax, 'register_network')
        assert hasattr(socialjax, 'get_network_class')
        assert hasattr(socialjax, 'list_networks')
        assert hasattr(socialjax, 'create_network')
        assert hasattr(socialjax, 'get_config_preset')
        assert hasattr(socialjax, 'list_config_presets')
        assert hasattr(socialjax, 'NETWORK_CONFIGS')
        assert hasattr(socialjax, 'CNNSmall')
        assert hasattr(socialjax, 'CNNActorCritic')
        assert hasattr(socialjax, 'NetworkAlreadyRegisteredError')
        assert hasattr(socialjax, 'NetworkNotFoundError')

    def test_buffer_exports(self):
        """Test buffer exports."""
        import socialjax
        assert hasattr(socialjax, 'BaseBuffer')
        assert hasattr(socialjax, 'RolloutBuffer')
        assert hasattr(socialjax, 'ReplayBuffer')
        assert hasattr(socialjax, 'PrioritizedReplayBuffer')
        assert hasattr(socialjax, 'BufferError')
        assert hasattr(socialjax, 'BufferEmptyError')
        assert hasattr(socialjax, 'BufferFullError')
        assert hasattr(socialjax, 'InsufficientDataError')

    def test_training_exports(self):
        """Test training exports."""
        import socialjax
        assert hasattr(socialjax, 'Trainer')
        assert hasattr(socialjax, 'create_trainer')
        assert hasattr(socialjax, 'BaseCallback')
        assert hasattr(socialjax, 'CallbackList')
        assert hasattr(socialjax, 'CheckpointCallback')
        assert hasattr(socialjax, 'EvalCallback')
        assert hasattr(socialjax, 'ProgressCallback')
        assert hasattr(socialjax, 'WandbCallback')

    def test_evaluation_exports(self):
        """Test evaluation exports."""
        import socialjax
        assert hasattr(socialjax, 'EpisodeMetrics')
        assert hasattr(socialjax, 'EvaluationMetrics')
        assert hasattr(socialjax, 'compute_episode_return')
        assert hasattr(socialjax, 'compute_cooperation_rate')
        assert hasattr(socialjax, 'Evaluator')
        assert hasattr(socialjax, 'EvaluatorConfig')
        assert hasattr(socialjax, 'save_gif')
        assert hasattr(socialjax, 'save_mp4')

    def test_config_exports(self):
        """Test configuration exports."""
        import socialjax
        assert hasattr(socialjax, 'TrainingConfig')
        assert hasattr(socialjax, 'NetworkConfig')
        assert hasattr(socialjax, 'AlgorithmConfig')
        assert hasattr(socialjax, 'EnvironmentConfig')
        assert hasattr(socialjax, 'SocialJaxConfig')
        assert hasattr(socialjax, 'ConfigManager')
        assert hasattr(socialjax, 'ConfigValidationError')
        assert hasattr(socialjax, 'create_default_config')


class TestQuickStartExample:
    """Test that Quick Start example code works."""

    def test_environment_creation(self):
        """Test environment creation example from docs."""
        import socialjax
        env = socialjax.make('clean_up', num_agents=7)
        assert env is not None
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')

    def test_get_algorithm(self):
        """Test getting algorithm class."""
        import socialjax
        AlgorithmClass = socialjax.get_algorithm('ippo')
        assert AlgorithmClass is not None

    def test_list_algorithms(self):
        """Test listing algorithms."""
        import socialjax
        algorithms = socialjax.list_algorithms()
        assert isinstance(algorithms, list)
        assert 'ippo' in algorithms
        assert 'mappo' in algorithms
        assert 'vdn' in algorithms
        assert 'svo' in algorithms


class TestEnvironmentCreation:
    """Test environment creation examples from docs."""

    def test_make_clean_up(self):
        """Test creating clean_up environment."""
        import socialjax
        env = socialjax.make('clean_up', num_agents=7)
        assert env is not None

    def test_registered_envs(self):
        """Test registered_envs is a list."""
        import socialjax
        assert isinstance(socialjax.registered_envs, list)
        assert 'clean_up' in socialjax.registered_envs


class TestAlgorithmExamples:
    """Test algorithm examples from docs."""

    def test_get_ippo_class(self):
        """Test getting IPPO class."""
        import socialjax
        IPPO = socialjax.get_algorithm('ippo')
        assert IPPO is not None

    def test_get_mappo_class(self):
        """Test getting MAPPO class."""
        import socialjax
        MAPPO = socialjax.get_algorithm('mappo')
        assert MAPPO is not None

    def test_get_vdn_class(self):
        """Test getting VDN class."""
        import socialjax
        VDN = socialjax.get_algorithm('vdn')
        assert VDN is not None

    def test_get_svo_class(self):
        """Test getting SVO class."""
        import socialjax
        SVO = socialjax.get_algorithm('svo')
        assert SVO is not None

    def test_algorithm_not_found_error(self):
        """Test AlgorithmNotFoundError is raised for unknown algorithms."""
        import socialjax
        with pytest.raises(socialjax.AlgorithmNotFoundError):
            socialjax.get_algorithm('unknown_algorithm')


class TestNetworkExamples:
    """Test network examples from docs."""

    def test_create_cnn_small(self):
        """Test creating CNN small network."""
        import socialjax
        network = socialjax.create_network('cnn_small', action_dim=8)
        assert network is not None

    def test_list_networks(self):
        """Test listing networks."""
        import socialjax
        networks = socialjax.list_networks()
        assert isinstance(networks, list)
        assert 'cnn_small' in networks

    def test_network_not_found_error(self):
        """Test NetworkNotFoundError is raised for unknown networks."""
        import socialjax
        with pytest.raises(socialjax.NetworkNotFoundError):
            socialjax.create_network('unknown_network', action_dim=8)

    def test_get_config_preset(self):
        """Test getting config preset."""
        import socialjax
        config = socialjax.get_config_preset('medium')
        assert isinstance(config, dict)


class TestCallbackExamples:
    """Test callback examples from docs."""

    def test_checkpoint_callback(self):
        """Test CheckpointCallback creation."""
        import socialjax
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = socialjax.CheckpointCallback(
                save_freq=1000,
                save_path=tmpdir,
                name_prefix='ippo',
                verbose=True,
            )
            assert callback is not None

    def test_eval_callback(self):
        """Test EvalCallback creation."""
        import socialjax
        env = socialjax.make('clean_up', num_agents=7)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = socialjax.EvalCallback(
                eval_env=env,
                eval_freq=5000,
                n_eval_episodes=10,
                best_model_save_path=tmpdir,
                deterministic=True,
            )
            assert callback is not None

    def test_progress_callback(self):
        """Test ProgressCallback creation."""
        import socialjax
        callback = socialjax.ProgressCallback(
            total_timesteps=1_000_000,
            progress_freq=10,
            verbose=True,
        )
        assert callback is not None

    def test_wandb_callback(self):
        """Test WandbCallback creation."""
        import socialjax
        callback = socialjax.WandbCallback(
            project='socialjax-test',
            name='test_run',
            log_freq=100,
        )
        assert callback is not None

    def test_callback_list(self):
        """Test CallbackList creation."""
        import socialjax
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            callbacks = socialjax.CallbackList([
                socialjax.CheckpointCallback(save_freq=1000, save_path=tmpdir),
                socialjax.ProgressCallback(total_timesteps=1000),
            ])
            assert len(callbacks) == 2


class TestBufferExamples:
    """Test buffer examples from docs."""

    def test_rollout_buffer(self):
        """Test RolloutBuffer creation."""
        import socialjax
        buffer = socialjax.RolloutBuffer(
            buffer_size=128,
            num_envs=8,
            obs_shape=(15, 15, 3),
            action_dim=8,
        )
        assert buffer is not None

    def test_replay_buffer(self):
        """Test ReplayBuffer creation."""
        import socialjax
        buffer = socialjax.ReplayBuffer(
            buffer_size=10000,
            obs_shape=(4,),
            action_dim=2,
        )
        assert buffer is not None

    def test_prioritized_replay_buffer(self):
        """Test PrioritizedReplayBuffer creation."""
        import socialjax
        buffer = socialjax.PrioritizedReplayBuffer(
            buffer_size=10000,
            obs_shape=(4,),
            action_dim=2,
        )
        assert buffer is not None


class TestConfigExamples:
    """Test configuration examples from docs."""

    def test_create_default_config(self):
        """Test create_default_config function."""
        import socialjax
        config = socialjax.create_default_config(
            algorithm='ippo',
            environment='clean_up',
        )
        assert config is not None

    def test_config_manager(self):
        """Test ConfigManager creation."""
        import socialjax
        manager = socialjax.ConfigManager()
        assert manager is not None

    def test_training_config(self):
        """Test TrainingConfig creation."""
        import socialjax
        config = socialjax.TrainingConfig(
            total_timesteps=1_000_000,
            num_envs=8,
            gamma=0.99,
            learning_rate=2.5e-4,
        )
        assert config is not None


class TestEvaluationExamples:
    """Test evaluation examples from docs."""

    def test_evaluator_config(self):
        """Test EvaluatorConfig creation."""
        import socialjax
        config = socialjax.EvaluatorConfig(
            num_episodes=50,
            deterministic=True,
            seed=42,
        )
        assert config is not None

    def test_episode_metrics(self):
        """Test EpisodeMetrics creation."""
        import socialjax
        metrics = socialjax.EpisodeMetrics(
            episode_return=100.0,
            episode_length=500,
            agent_returns={'agent_0': 50.0, 'agent_1': 50.0},
        )
        assert metrics is not None


class TestDocstringVerification:
    """Test that all public classes have docstrings."""

    def test_base_algorithm_docstring(self):
        """Test BaseAlgorithm has docstring."""
        import socialjax
        assert socialjax.BaseAlgorithm.__doc__ is not None
        assert len(socialjax.BaseAlgorithm.__doc__) > 50

    def test_base_trainer_docstring(self):
        """Test BaseTrainer has docstring."""
        import socialjax
        assert socialjax.BaseTrainer.__doc__ is not None
        assert len(socialjax.BaseTrainer.__doc__) > 50

    def test_trainer_docstring(self):
        """Test Trainer has docstring."""
        import socialjax
        assert socialjax.Trainer.__doc__ is not None
        assert len(socialjax.Trainer.__doc__) > 50

    def test_base_callback_docstring(self):
        """Test BaseCallback has docstring."""
        import socialjax
        assert socialjax.BaseCallback.__doc__ is not None
        assert len(socialjax.BaseCallback.__doc__) > 50

    def test_config_manager_docstring(self):
        """Test ConfigManager has docstring."""
        import socialjax
        assert socialjax.ConfigManager.__doc__ is not None
        assert len(socialjax.ConfigManager.__doc__) > 50

    def test_evaluator_docstring(self):
        """Test Evaluator has docstring."""
        import socialjax
        assert socialjax.Evaluator.__doc__ is not None
        assert len(socialjax.Evaluator.__doc__) > 50


class TestAPICallable:
    """Test that API functions are callable."""

    def test_make_callable(self):
        """Test make is callable."""
        import socialjax
        assert callable(socialjax.make)

    def test_get_algorithm_callable(self):
        """Test get_algorithm is callable."""
        import socialjax
        assert callable(socialjax.get_algorithm)

    def test_list_algorithms_callable(self):
        """Test list_algorithms is callable."""
        import socialjax
        assert callable(socialjax.list_algorithms)

    def test_create_network_callable(self):
        """Test create_network is callable."""
        import socialjax
        assert callable(socialjax.create_network)

    def test_list_networks_callable(self):
        """Test list_networks is callable."""
        import socialjax
        assert callable(socialjax.list_networks)

    def test_create_default_config_callable(self):
        """Test create_default_config is callable."""
        import socialjax
        assert callable(socialjax.create_default_config)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
