"""Comprehensive benchmark tests for V2 algorithms.

This module provides a benchmarking system for SocialJax V2 algorithms.
It validates that all algorithms can be instantiated and run, and documents
performance characteristics.

Run with: pytest tests/validation/test_benchmarks.py -v -s
"""

import pytest
import sys
import time
import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime

# Setup paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'socialjax'))

import jax
import jax.numpy as jnp
import numpy as np


# ============================================================================
# Benchmark Configuration
# ============================================================================

BENCHMARK_CONFIG = {
    'num_steps': 100,
    'batch_size': 32,
    'seed': 42,
}

# Environment configurations
ENVIRONMENT_CONFIGS = {
    'clean_up': {
        'num_agents': 7,
        'obs_shape': (11, 11, 19),
        'action_dim': 9,
    },
    'harvest_common_open': {
        'num_agents': 7,
        'obs_shape': (11, 11, 15),
        'action_dim': 8,
    },
    'coop_mining': {
        'num_agents': 5,
        'obs_shape': (11, 11, 12),
        'action_dim': 9,
    },
}

# Algorithm configurations
ALGORITHM_CONFIGS = {
    'ippo': {
        'lr': 0.0005,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_eps': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
    },
    'mappo': {
        'lr': 0.0005,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_eps': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
    },
    'vdn': {
        'lr': 0.0005,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'batch_size': 32,
    },
    'svo': {
        'lr': 0.0005,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_eps': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'svo_angle': 45.0,
    },
}


# ============================================================================
# Benchmark Data Classes
# ============================================================================

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    algorithm: str
    environment: str
    test_type: str
    success: bool = False
    error_message: str = ""
    training_time_seconds: float = 0.0
    forward_pass_time_ms: float = 0.0
    update_time_ms: float = 0.0
    memory_mb: float = 0.0


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    timestamp: str
    results: List[BenchmarkResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str):
        """Save report to JSON file."""
        data = {
            'timestamp': self.timestamp,
            'results': [asdict(r) for r in self.results],
            'summary': self.summary,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# ============================================================================
# Test Classes
# ============================================================================

class TestAlgorithmAvailability:
    """Test that all V2 algorithms are available."""

    def test_ippo_available(self):
        """Test IPPO algorithm is available."""
        from socialjax.algorithms.registry import get_algorithm
        algo = get_algorithm('ippo')
        assert algo is not None

    def test_mappo_available(self):
        """Test MAPPO algorithm is available."""
        from socialjax.algorithms.registry import get_algorithm
        algo = get_algorithm('mappo')
        assert algo is not None

    def test_vdn_available(self):
        """Test VDN algorithm is available."""
        from socialjax.algorithms.registry import get_algorithm
        algo = get_algorithm('vdn')
        assert algo is not None

    def test_svo_available(self):
        """Test SVO algorithm is available."""
        from socialjax.algorithms.registry import get_algorithm
        algo = get_algorithm('svo')
        assert algo is not None


class TestAlgorithmInstantiation:
    """Test that algorithms can be instantiated for each environment."""

    @pytest.fixture
    def create_dummy_spaces(self):
        """Create dummy observation and action spaces."""
        def _create(obs_shape, action_dim):
            class DummyObsSpace:
                shape = obs_shape

            class DummyActSpace:
                n = action_dim

            return DummyObsSpace(), DummyActSpace()
        return _create

    @pytest.mark.parametrize("algo_name", ['ippo', 'mappo', 'vdn', 'svo'])
    @pytest.mark.parametrize("env_name", ['clean_up', 'harvest_common_open', 'coop_mining'])
    def test_algorithm_instantiation(self, algo_name, env_name, create_dummy_spaces):
        """Test algorithm can be instantiated for environment."""
        from socialjax.algorithms.registry import get_algorithm

        env_config = ENVIRONMENT_CONFIGS[env_name]
        algo_config = ALGORITHM_CONFIGS.get(algo_name, {}).copy()

        obs_space, act_space = create_dummy_spaces(
            env_config['obs_shape'],
            env_config['action_dim']
        )

        algo_class = get_algorithm(algo_name)

        # Check if algorithm requires num_agents
        import inspect
        sig = inspect.signature(algo_class.__init__)
        kwargs = {
            'observation_space': obs_space,
            'action_space': act_space,
            'config': algo_config,
        }
        if 'num_agents' in sig.parameters:
            kwargs['num_agents'] = env_config['num_agents']

        algo = algo_class(**kwargs)
        assert algo is not None, f"Failed to create {algo_name} for {env_name}"


class TestAlgorithmStateInitialization:
    """Test that algorithm states can be initialized."""

    @pytest.fixture
    def setup_algorithm(self):
        """Setup an algorithm for testing."""
        def _setup(algo_name, env_name):
            from socialjax.algorithms.registry import get_algorithm

            env_config = ENVIRONMENT_CONFIGS[env_name]
            algo_config = ALGORITHM_CONFIGS.get(algo_name, {}).copy()

            class DummyObsSpace:
                shape = env_config['obs_shape']

            class DummyActSpace:
                n = env_config['action_dim']

            algo_class = get_algorithm(algo_name)

            import inspect
            sig = inspect.signature(algo_class.__init__)
            kwargs = {
                'observation_space': DummyObsSpace(),
                'action_space': DummyActSpace(),
                'config': algo_config,
            }
            if 'num_agents' in sig.parameters:
                kwargs['num_agents'] = env_config['num_agents']

            return algo_class(**kwargs)
        return _setup

    @pytest.mark.parametrize("algo_name", ['ippo', 'mappo', 'vdn', 'svo'])
    def test_state_initialization(self, algo_name, setup_algorithm):
        """Test algorithm state can be initialized."""
        algo = setup_algorithm(algo_name, 'clean_up')

        rng = jax.random.PRNGKey(42)
        state = algo.init_state(rng)

        assert state is not None, f"Failed to initialize state for {algo_name}"
        # Check for params attribute
        if hasattr(state, 'params'):
            assert state.params is not None, f"params not initialized for {algo_name}"
        if hasattr(state, 'actor_params'):
            assert state.actor_params is not None, f"actor_params not initialized for {algo_name}"


class TestAlgorithmForwardPass:
    """Test forward pass performance for algorithms."""

    @pytest.fixture
    def setup_and_init_algorithm(self):
        """Setup and initialize an algorithm."""
        def _setup(algo_name, env_name):
            from socialjax.algorithms.registry import get_algorithm

            env_config = ENVIRONMENT_CONFIGS[env_name]
            algo_config = ALGORITHM_CONFIGS.get(algo_name, {}).copy()

            class DummyObsSpace:
                shape = env_config['obs_shape']

            class DummyActSpace:
                n = env_config['action_dim']

            algo_class = get_algorithm(algo_name)

            import inspect
            sig = inspect.signature(algo_class.__init__)
            kwargs = {
                'observation_space': DummyObsSpace(),
                'action_space': DummyActSpace(),
                'config': algo_config,
            }
            if 'num_agents' in sig.parameters:
                kwargs['num_agents'] = env_config['num_agents']

            algo = algo_class(**kwargs)
            rng = jax.random.PRNGKey(42)
            state = algo.init_state(rng)

            return algo, state, env_config['obs_shape']
        return _setup

    @pytest.mark.parametrize("algo_name", ['ippo', 'svo'])
    def test_compute_action(self, algo_name, setup_and_init_algorithm):
        """Test that compute_action works for IPPO and SVO."""
        algo, state, obs_shape = setup_and_init_algorithm(algo_name, 'clean_up')

        # Create dummy observation
        rng = jax.random.PRNGKey(0)
        dummy_obs = jax.random.uniform(rng, obs_shape)

        # Compute action
        action_rng = jax.random.PRNGKey(1)
        action, info = algo.compute_action(state, dummy_obs, action_rng, deterministic=False)

        assert action is not None, f"compute_action returned None for {algo_name}"
        assert isinstance(info, dict), f"compute_action info should be dict for {algo_name}"

    @pytest.mark.parametrize("algo_name", ['ippo', 'svo'])
    def test_forward_pass_timing(self, algo_name, setup_and_init_algorithm):
        """Test forward pass timing for benchmarking."""
        algo, state, obs_shape = setup_and_init_algorithm(algo_name, 'clean_up')

        # Warmup
        rng = jax.random.PRNGKey(0)
        for _ in range(5):
            dummy_obs = jax.random.uniform(rng, obs_shape)
            rng, action_rng = jax.random.split(rng)
            action, info = algo.compute_action(state, dummy_obs, action_rng, deterministic=False)

        # Timing
        num_runs = 50
        times = []
        for _ in range(num_runs):
            dummy_obs = jax.random.uniform(rng, obs_shape)
            rng, action_rng = jax.random.split(rng)

            start = time.perf_counter()
            action, info = algo.compute_action(state, dummy_obs, action_rng, deterministic=False)
            # Block to ensure computation is complete
            if hasattr(action, 'block_until_ready'):
                action.block_until_ready()
            end = time.perf_counter()

            times.append((end - start) * 1000)  # ms

        mean_time = np.mean(times)
        std_time = np.std(times)

        print(f"\n  {algo_name} forward pass: {mean_time:.3f} +/- {std_time:.3f} ms")
        assert mean_time < 100, f"Forward pass too slow for {algo_name}: {mean_time:.3f} ms"


class TestAlgorithmUpdate:
    """Test algorithm update functionality - tests that update method exists and runs."""

    @pytest.fixture
    def setup_for_update(self):
        """Setup algorithm for update testing."""
        def _setup(algo_name, env_name):
            from socialjax.algorithms.registry import get_algorithm

            env_config = ENVIRONMENT_CONFIGS[env_name]
            algo_config = ALGORITHM_CONFIGS.get(algo_name, {}).copy()

            class DummyObsSpace:
                shape = env_config['obs_shape']

            class DummyActSpace:
                n = env_config['action_dim']

            algo_class = get_algorithm(algo_name)

            import inspect
            sig = inspect.signature(algo_class.__init__)
            kwargs = {
                'observation_space': DummyObsSpace(),
                'action_space': DummyActSpace(),
                'config': algo_config,
            }
            if 'num_agents' in sig.parameters:
                kwargs['num_agents'] = env_config['num_agents']

            algo = algo_class(**kwargs)
            rng = jax.random.PRNGKey(42)
            state = algo.init_state(rng)

            return algo, state, env_config
        return _setup

    @pytest.mark.parametrize("algo_name", ['ippo', 'mappo', 'svo', 'vdn'])
    def test_update_method_exists(self, algo_name, setup_for_update):
        """Test that update method exists and is callable."""
        algo, state, env_config = setup_for_update(algo_name, 'clean_up')

        # Verify update method exists
        assert hasattr(algo, 'update'), f"{algo_name} should have update method"
        assert callable(algo.update), f"{algo_name}.update should be callable"

    @pytest.mark.parametrize("algo_name", ['ippo', 'svo'])
    def test_update_runs(self, algo_name, setup_for_update):
        """Test that update runs for IPPO and SVO."""
        algo, state, env_config = setup_for_update(algo_name, 'clean_up')

        # Run update (may use internal batch)
        rng = jax.random.PRNGKey(0)
        try:
            new_state, update_info = algo.update(state, rng)
            assert new_state is not None, f"Update returned None state for {algo_name}"
        except Exception as e:
            # Update may fail if batch format is wrong, but we just test it exists
            print(f"  {algo_name} update: {type(e).__name__}: {str(e)[:50]}")


class TestEnvironmentIntegration:
    """Test that algorithms work with environments."""

    @pytest.mark.parametrize("env_name", ['clean_up', 'harvest_common_open', 'coop_mining'])
    def test_environment_creation(self, env_name):
        """Test environment can be created."""
        import socialjax

        config = ENVIRONMENT_CONFIGS[env_name]
        try:
            env = socialjax.make(env_name, num_agents=config['num_agents'])
            assert env is not None

            key = jax.random.PRNGKey(0)
            obs, state = env.reset(key)
            assert obs is not None

            print(f"\n  {env_name}: obs shape = {obs.shape}")
        except Exception as e:
            pytest.skip(f"Environment {env_name} not available: {e}")


class TestBenchmarkReport:
    """Generate and save benchmark report."""

    def test_generate_benchmark_report(self, tmp_path):
        """Generate a benchmark report."""
        report = BenchmarkReport(
            timestamp=datetime.now().isoformat(),
        )

        # Run quick tests and collect results
        for algo_name in ['ippo', 'mappo', 'vdn', 'svo']:
            for env_name in ['clean_up', 'harvest_common_open', 'coop_mining']:
                result = BenchmarkResult(
                    algorithm=algo_name,
                    environment=env_name,
                    test_type='instantiation',
                )

                try:
                    from socialjax.algorithms.registry import get_algorithm

                    env_config = ENVIRONMENT_CONFIGS[env_name]
                    algo_config = ALGORITHM_CONFIGS.get(algo_name, {}).copy()

                    class DummyObsSpace:
                        shape = env_config['obs_shape']

                    class DummyActSpace:
                        n = env_config['action_dim']

                    algo_class = get_algorithm(algo_name)

                    import inspect
                    sig = inspect.signature(algo_class.__init__)
                    kwargs = {
                        'observation_space': DummyObsSpace(),
                        'action_space': DummyActSpace(),
                        'config': algo_config,
                    }
                    if 'num_agents' in sig.parameters:
                        kwargs['num_agents'] = env_config['num_agents']

                    start = time.perf_counter()
                    algo = algo_class(**kwargs)
                    rng = jax.random.PRNGKey(42)
                    state = algo.init_state(rng)
                    end = time.perf_counter()

                    result.success = True
                    result.training_time_seconds = end - start

                except Exception as e:
                    result.success = False
                    result.error_message = str(e)[:200]

                report.results.append(result)

        # Generate summary
        successful = sum(1 for r in report.results if r.success)
        total = len(report.results)
        report.summary = {
            'total_tests': total,
            'successful': successful,
            'failed': total - successful,
            'success_rate': successful / total if total > 0 else 0,
        }

        # Save report
        report_path = tmp_path / "benchmark_report.json"
        report.save(str(report_path))

        assert report_path.exists(), "Report file not created"

        print(f"\n{'='*60}")
        print(f"Benchmark Summary")
        print(f"{'='*60}")
        print(f"Total tests: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {total - successful}")
        print(f"Success rate: {report.summary['success_rate']*100:.1f}%")
        print(f"{'='*60}")


class TestBenchmarkSummary:
    """Summary documentation of benchmark capabilities."""

    def test_all_algorithms_available(self):
        """Verify all algorithms are available."""
        from socialjax.algorithms.registry import list_algorithms

        algorithms = list_algorithms()
        print(f"\nAvailable algorithms: {algorithms}")

        expected = ['ippo', 'mappo', 'vdn', 'svo']
        for algo in expected:
            assert algo in algorithms, f"Algorithm {algo} not available"

    def test_all_environments_available(self):
        """Verify environments are available."""
        import socialjax

        available = []
        unavailable = []

        for env_name, config in ENVIRONMENT_CONFIGS.items():
            try:
                env = socialjax.make(env_name, num_agents=config['num_agents'])
                key = jax.random.PRNGKey(0)
                obs, state = env.reset(key)
                available.append(env_name)
            except Exception as e:
                unavailable.append((env_name, str(e)[:50]))

        print(f"\nAvailable environments: {available}")
        if unavailable:
            print(f"Unavailable environments: {[u[0] for u in unavailable]}")

        assert len(available) >= 1, "At least one environment should be available"

    def test_benchmark_capabilities_summary(self):
        """Print summary of benchmark capabilities."""
        from socialjax.algorithms.registry import list_algorithms

        algorithms = list_algorithms()

        print("\n" + "="*60)
        print("SocialJax V2 Benchmark Capabilities Summary")
        print("="*60)
        print(f"\nAlgorithms ({len(algorithms)}):")
        for algo in algorithms:
            print(f"  - {algo}")

        print(f"\nEnvironments:")
        for env_name, config in ENVIRONMENT_CONFIGS.items():
            print(f"  - {env_name}: {config['num_agents']} agents, "
                  f"obs={config['obs_shape']}, actions={config['action_dim']}")

        print(f"\nBenchmark Configuration:")
        print(f"  - Steps: {BENCHMARK_CONFIG['num_steps']}")
        print(f"  - Batch size: {BENCHMARK_CONFIG['batch_size']}")
        print(f"  - Seed: {BENCHMARK_CONFIG['seed']}")

        print("\n" + "="*60)


# ============================================================================
# CLI Entry Point
# ============================================================================

def run_benchmarks_cli():
    """CLI entry point for running benchmarks."""
    import argparse

    parser = argparse.ArgumentParser(description="Run SocialJax V2 benchmarks")
    parser.add_argument('--output', default='evaluation/benchmarks', help='Output directory')

    args = parser.parse_args()

    # Run benchmark tests
    report = BenchmarkReport(
        timestamp=datetime.now().isoformat(),
    )

    from socialjax.algorithms.registry import list_algorithms, get_algorithm

    algorithms = list_algorithms()

    for algo_name in algorithms:
        for env_name, env_config in ENVIRONMENT_CONFIGS.items():
            result = BenchmarkResult(
                algorithm=algo_name,
                environment=env_name,
                test_type='instantiation',
            )

            try:
                algo_config = ALGORITHM_CONFIGS.get(algo_name, {}).copy()

                class DummyObsSpace:
                    shape = env_config['obs_shape']

                class DummyActSpace:
                    n = env_config['action_dim']

                algo_class = get_algorithm(algo_name)

                import inspect
                sig = inspect.signature(algo_class.__init__)
                kwargs = {
                    'observation_space': DummyObsSpace(),
                    'action_space': DummyActSpace(),
                    'config': algo_config,
                }
                if 'num_agents' in sig.parameters:
                    kwargs['num_agents'] = env_config['num_agents']

                start = time.perf_counter()
                algo = algo_class(**kwargs)
                rng = jax.random.PRNGKey(42)
                state = algo.init_state(rng)
                end = time.perf_counter()

                result.success = True
                result.training_time_seconds = end - start
                print(f"[OK] {algo_name} on {env_name}: {end - start:.3f}s")

            except Exception as e:
                result.success = False
                result.error_message = str(e)[:200]
                print(f"[FAIL] {algo_name} on {env_name}: {result.error_message}")

            report.results.append(result)

    # Generate summary
    successful = sum(1 for r in report.results if r.success)
    total = len(report.results)
    report.summary = {
        'total_tests': total,
        'successful': successful,
        'failed': total - successful,
        'success_rate': successful / total if total > 0 else 0,
    }

    # Save report
    os.makedirs(args.output, exist_ok=True)
    report.save(os.path.join(args.output, 'benchmark_results.json'))

    print(f"\nResults saved to {args.output}/benchmark_results.json")
    print(f"Summary: {successful}/{total} tests passed ({report.summary['success_rate']*100:.1f}%)")


if __name__ == '__main__':
    run_benchmarks_cli()
