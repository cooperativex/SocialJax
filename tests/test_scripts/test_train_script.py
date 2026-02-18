"""Unit tests for scripts/train.py.

Tests cover:
- CLI argument parsing
- Configuration loading
- Callback building
- Signal handling
- Main entry point behavior
"""

import pytest
import sys
import os
import signal
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import io

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "socialjax"))

# Import the module under test
from scripts.train import (
    parse_args,
    load_config,
    build_callbacks,
    signal_handler,
    format_time,
    print_training_info,
)


class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_required_args_algorithm_and_env(self):
        """Test that algorithm and env are required."""
        # Missing both args
        with patch('sys.argv', ['train.py']):
            with pytest.raises(SystemExit):
                parse_args()

        # Missing env
        with patch('sys.argv', ['train.py', '--algorithm', 'ippo']):
            with pytest.raises(SystemExit):
                parse_args()

        # Missing algorithm
        with patch('sys.argv', ['train.py', '--env', 'coin_game']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_valid_minimal_args(self):
        """Test minimal valid argument set."""
        with patch('sys.argv', ['train.py', '--algorithm', 'ippo', '--env', 'coin_game']):
            args = parse_args()
            assert args.algorithm == 'ippo'
            assert args.env == 'coin_game'
            assert args.timesteps == 1_000_000
            assert args.seed == 42

    def test_algorithm_choices(self):
        """Test that algorithm must be from valid choices."""
        with patch('sys.argv', ['train.py', '--algorithm', 'invalid', '--env', 'coin_game']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_custom_timesteps(self):
        """Test custom timesteps argument."""
        with patch('sys.argv', ['train.py', '--algorithm', 'ippo', '--env', 'coin_game', '--timesteps', '500000']):
            args = parse_args()
            assert args.timesteps == 500000

    def test_custom_seed(self):
        """Test custom seed argument."""
        with patch('sys.argv', ['train.py', '--algorithm', 'ippo', '--env', 'coin_game', '--seed', '123']):
            args = parse_args()
            assert args.seed == 123

    def test_wandb_options(self):
        """Test WandB arguments."""
        with patch('sys.argv', [
            'train.py',
            '--algorithm', 'ippo',
            '--env', 'coin_game',
            '--wandb-project', 'myproject',
            '--wandb-name', 'experiment1',
            '--wandb-entity', 'myteam'
        ]):
            args = parse_args()
            assert args.wandb_project == 'myproject'
            assert args.wandb_name == 'experiment1'
            assert args.wandb_entity == 'myteam'

    def test_config_file_option(self):
        """Test config file argument."""
        with patch('sys.argv', ['train.py', '--algorithm', 'ippo', '--env', 'coin_game', '--config', 'myconfig.yaml']):
            args = parse_args()
            assert args.config == 'myconfig.yaml'

    def test_learning_rate_override(self):
        """Test learning rate CLI override."""
        with patch('sys.argv', ['train.py', '--algorithm', 'ippo', '--env', 'coin_game', '--lr', '0.001']):
            args = parse_args()
            assert args.lr == 0.001

    def test_gamma_override(self):
        """Test gamma CLI override."""
        with patch('sys.argv', ['train.py', '--algorithm', 'ippo', '--env', 'coin_game', '--gamma', '0.95']):
            args = parse_args()
            assert args.gamma == 0.95

    def test_checkpoint_options(self):
        """Test checkpoint arguments."""
        with patch('sys.argv', [
            'train.py',
            '--algorithm', 'ippo',
            '--env', 'coin_game',
            '--checkpoint-dir', 'my_checkpoints',
            '--checkpoint-freq', '5000',
            '--save-best'
        ]):
            args = parse_args()
            assert args.checkpoint_dir == 'my_checkpoints'
            assert args.checkpoint_freq == 5000
            assert args.save_best is True

    def test_verbose_option(self):
        """Test verbosity level."""
        with patch('sys.argv', ['train.py', '--algorithm', 'ippo', '--env', 'coin_game', '--verbose', '2']):
            args = parse_args()
            assert args.verbose == 2

    def test_no_progress_flag(self):
        """Test no-progress flag."""
        with patch('sys.argv', ['train.py', '--algorithm', 'ippo', '--env', 'coin_game', '--no-progress']):
            args = parse_args()
            assert args.no_progress is True


class TestLoadConfig:
    """Tests for configuration loading."""

    @pytest.fixture
    def mock_args(self):
        """Create mock args for testing."""
        args = MagicMock()
        args.algorithm = 'ippo'
        args.env = 'coin_game'
        args.config = None
        args.verbose = 1
        args.lr = None
        args.gamma = None
        args.gae_lambda = None
        args.timesteps = 1_000_000
        args.num_envs = 1
        args.num_steps = 128
        args.seed = 42
        return args

    def test_load_config_returns_dict(self, mock_args):
        """Test that load_config returns a dictionary."""
        config = load_config(mock_args)
        assert isinstance(config, dict)

    def test_load_config_includes_algorithm(self, mock_args):
        """Test that config includes algorithm name."""
        config = load_config(mock_args)
        assert 'algorithm' in config
        assert config['algorithm']['name'] == 'ippo'

    def test_load_config_includes_environment(self, mock_args):
        """Test that config includes environment name."""
        config = load_config(mock_args)
        assert 'environment' in config
        assert config['environment']['name'] == 'coin_game'

    def test_load_config_applies_timesteps_override(self, mock_args):
        """Test that timesteps from args is applied to config."""
        mock_args.timesteps = 500000
        config = load_config(mock_args)
        assert config['algorithm']['training']['total_timesteps'] == 500000

    def test_load_config_applies_lr_override(self, mock_args):
        """Test that learning rate override is applied."""
        mock_args.lr = 0.001
        config = load_config(mock_args)
        assert config['algorithm']['training']['learning_rate'] == 0.001

    def test_load_config_applies_seed(self, mock_args):
        """Test that seed is applied to config."""
        mock_args.seed = 123
        config = load_config(mock_args)
        assert config['algorithm']['training']['seed'] == 123


class TestBuildCallbacks:
    """Tests for callback building."""

    @pytest.fixture
    def mock_args(self):
        """Create mock args for testing."""
        args = MagicMock()
        args.algorithm = 'ippo'
        args.env = 'coin_game'
        args.checkpoint_dir = 'checkpoints'
        args.checkpoint_freq = 10000
        args.verbose = 1
        args.wandb_project = None
        args.wandb_name = None
        args.wandb_entity = None
        return args

    def test_build_callbacks_returns_callback_list(self, mock_args):
        """Test that build_callbacks returns a CallbackList."""
        from socialjax.training.callbacks import CallbackList
        callbacks = build_callbacks(mock_args)
        assert isinstance(callbacks, CallbackList)

    def test_checkpoint_callback_always_included(self, mock_args):
        """Test that CheckpointCallback is always included."""
        from socialjax.training.callbacks import CheckpointCallback
        callbacks = build_callbacks(mock_args)
        assert len(callbacks) >= 1
        # First callback should be CheckpointCallback
        assert isinstance(callbacks.callbacks[0], CheckpointCallback)

    def test_wandb_callback_included_when_project_set(self, mock_args):
        """Test that WandbCallback is included when project is set."""
        from socialjax.training.callbacks import WandbCallback
        mock_args.wandb_project = 'myproject'
        callbacks = build_callbacks(mock_args)
        # Should have CheckpointCallback + WandbCallback
        assert len(callbacks) >= 2
        wandb_found = any(isinstance(cb, WandbCallback) for cb in callbacks.callbacks)
        assert wandb_found

    def test_checkpoint_path_includes_algorithm_and_env(self, mock_args):
        """Test that checkpoint path includes algorithm and env names."""
        mock_args.checkpoint_dir = 'my_checkpoints'
        callbacks = build_callbacks(mock_args)
        checkpoint_callback = callbacks.callbacks[0]
        assert 'ippo' in checkpoint_callback.save_path
        assert 'coin_game' in checkpoint_callback.save_path


class TestSignalHandler:
    """Tests for signal handling."""

    def test_signal_handler_sets_interrupted_flag(self):
        """Test that signal handler sets the global _interrupted flag."""
        import scripts.train as train_module
        train_module._interrupted = False

        signal_handler(signal.SIGINT, None)
        assert train_module._interrupted is True

        # Reset
        train_module._interrupted = False


class TestFormatTime:
    """Tests for time formatting utility."""

    def test_format_seconds(self):
        """Test formatting seconds."""
        assert format_time(30.0) == "30.0s"
        assert format_time(59.9) == "59.9s"

    def test_format_minutes(self):
        """Test formatting minutes."""
        assert format_time(60.0) == "1.0m"
        assert format_time(120.0) == "2.0m"
        assert format_time(1800.0) == "30.0m"

    def test_format_hours(self):
        """Test formatting hours."""
        assert format_time(3600.0) == "1.0h"
        assert format_time(7200.0) == "2.0h"


class TestPrintTrainingInfo:
    """Tests for training info printing."""

    def test_print_training_info_outputs_info(self, capsys):
        """Test that print_training_info prints configuration."""
        from types import SimpleNamespace

        args = SimpleNamespace(
            algorithm='ippo',
            env='coin_game',
            timesteps=1000000,
            seed=42,
            num_envs=1,
            num_steps=128,
            lr=None,
            gamma=None,
            gae_lambda=None,
            checkpoint_dir='checkpoints',
            checkpoint_freq=10000,
            wandb_project=None,
            wandb_name=None,
        )

        config = {
            'algorithm': {'name': 'ippo'},
            'environment': {'name': 'coin_game'},
        }

        print_training_info(args, config)

        captured = capsys.readouterr()
        assert 'ippo' in captured.out
        assert 'coin_game' in captured.out
        assert '1,000,000' in captured.out or '1000000' in captured.out


class TestTrainingExecution:
    """Integration-style tests for training execution."""

    @pytest.fixture
    def mock_args_full(self):
        """Create complete mock args for testing."""
        args = MagicMock()
        args.algorithm = 'ippo'
        args.env = 'coin_game'
        args.config = None
        args.verbose = 0  # Silent mode for tests
        args.timesteps = 1000  # Small for testing
        args.seed = 42
        args.num_envs = 1
        args.num_steps = 10
        args.lr = None
        args.gamma = None
        args.gae_lambda = None
        args.checkpoint_dir = tempfile.mkdtemp()
        args.checkpoint_freq = 100
        args.save_best = False
        args.wandb_project = None
        args.wandb_name = None
        args.wandb_entity = None
        args.eval_freq = 100
        args.eval_episodes = 1
        args.no_progress = True
        return args

    def test_config_loading_with_mock_args(self, mock_args_full):
        """Test that config loading works with mock args."""
        config = load_config(mock_args_full)
        assert isinstance(config, dict)
        assert 'algorithm' in config
        assert 'environment' in config

    def test_callbacks_building_with_mock_args(self, mock_args_full):
        """Test that callbacks can be built with mock args."""
        from socialjax.training.callbacks import CallbackList
        callbacks = build_callbacks(mock_args_full)
        assert isinstance(callbacks, CallbackList)
        assert len(callbacks) >= 1


class TestCLIIntegration:
    """Tests for CLI integration."""

    def test_help_command_works(self):
        """Test that --help works without error."""
        result = os.system(
            f'cd {project_root} && '
            f'PYTHONPATH={project_root}/socialjax:$PYTHONPATH '
            f'python scripts/train.py --help > /dev/null 2>&1'
        )
        assert result == 0

    def test_missing_required_args_exits(self):
        """Test that missing required args causes exit."""
        result = os.system(
            f'cd {project_root} && '
            f'PYTHONPATH={project_root}/socialjax:$PYTHONPATH '
            f'python scripts/train.py > /dev/null 2>&1'
        )
        assert result != 0  # Should fail

    def test_invalid_algorithm_exits(self):
        """Test that invalid algorithm causes exit."""
        result = os.system(
            f'cd {project_root} && '
            f'PYTHONPATH={project_root}/socialjax:$PYTHONPATH '
            f'python scripts/train.py --algorithm invalid_algo --env coin_game > /dev/null 2>&1'
        )
        assert result != 0  # Should fail


class TestKeyboardInterrupt:
    """Tests for keyboard interrupt handling."""

    def test_signal_handler_sets_flag(self):
        """Test that signal handler properly sets interrupted flag."""
        import scripts.train as train_module

        # Reset flag
        train_module._interrupted = False

        # Call handler
        signal_handler(signal.SIGINT, None)

        # Check flag
        assert train_module._interrupted is True

        # Reset for other tests
        train_module._interrupted = False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
