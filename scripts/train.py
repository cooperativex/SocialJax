#!/usr/bin/env python
"""Unified training script for SocialJax.

This script provides a command-line interface for training RL algorithms
on multi-agent environments.

Example usage:
    # Train IPPO on coin_game with default settings
    python scripts/train.py --algorithm ippo --env coin_game

    # Train with custom config file
    python scripts/train.py --algorithm ippo --env coin_game --config configs/ippo_coins.yaml

    # Train with WandB logging
    python scripts/train.py --algorithm ippo --env coin_game --wandb-project socialjax --wandb-name experiment1

    # Train with custom timesteps and seed
    python scripts/train.py --algorithm mappo --env clean_up --timesteps 1000000 --seed 42
"""

import argparse
import sys
import os
import signal
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))  # Add project root for socialjax module
sys.path.insert(0, str(project_root / "socialjax"))  # Also add socialjax for direct imports

import jax

from socialjax.training import Trainer, create_trainer
from socialjax.training.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    WandbCallback,
)
from socialjax.config.manager import ConfigManager, SocialJaxConfig
from socialjax.algorithms.registry import list_algorithms

# Import algorithm modules to register them
try:
    import socialjax.algorithms.ippo
except ImportError:
    pass
try:
    import socialjax.algorithms.mappo
except ImportError:
    pass
try:
    import socialjax.algorithms.vdn
except ImportError:
    pass
try:
    import socialjax.algorithms.svo
except ImportError:
    pass


# Global flag for graceful interrupt handling
_interrupted = False


def signal_handler(signum, frame):
    """Handle keyboard interrupt signals gracefully."""
    global _interrupted
    _interrupted = True
    print("\n\nInterrupt received! Saving checkpoint before exit...")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Train a SocialJax RL algorithm on a multi-agent environment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train.py --algorithm ippo --env coin_game
  python scripts/train.py --algorithm mappo --env clean_up --timesteps 1000000
  python scripts/train.py --algorithm ippo --env coin_game --config custom.yaml
  python scripts/train.py --algorithm ippo --env coin_game --wandb-project myproject
        """,
    )

    # Required arguments
    # Get available algorithms dynamically
    available_algorithms = list_algorithms()
    # If no algorithms registered yet, provide common defaults
    if not available_algorithms:
        available_algorithms = ["ippo", "mappo", "vdn", "svo"]

    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=available_algorithms,
        help=f"Algorithm to train (available: {', '.join(available_algorithms)})",
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Environment name (e.g., coin_game, clean_up, harvest_common_open)",
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config file (YAML)",
    )

    # Training parameters
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total timesteps to train (default: 1000000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments (default: 1)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=128,
        help="Steps per rollout (default: 128)",
    )

    # Learning parameters
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Discount factor (overrides config)",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=None,
        help="GAE lambda parameter (overrides config)",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10000,
        help="Save checkpoint every N steps (default: 10000)",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="Save best model during training",
    )

    # WandB logging
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="WandB project name (enables WandB logging)",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="WandB run name (default: algorithm_env_timestamp)",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB entity/team name",
    )

    # Evaluation
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10000,
        help="Evaluation frequency (default: 10000)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)",
    )

    # Misc
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level: 0=silent, 1=normal, 2=debug (default: 1)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress display",
    )

    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Load configuration from file and CLI arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Configuration dictionary
    """
    config_manager = ConfigManager()

    if args.config and os.path.exists(args.config):
        # Load from file
        if args.verbose >= 1:
            print(f"Loading config from: {args.config}")
        config = config_manager.load_from_file(args.config)
    else:
        # Create default config
        config = config_manager.load(args.algorithm, args.env)

    # Apply CLI overrides
    overrides = {}

    if args.lr is not None:
        overrides.setdefault("algorithm", {}).setdefault("training", {})["learning_rate"] = args.lr
    if args.gamma is not None:
        overrides.setdefault("algorithm", {}).setdefault("training", {})["gamma"] = args.gamma
    if args.gae_lambda is not None:
        overrides.setdefault("algorithm", {}).setdefault("training", {})["gae_lambda"] = args.gae_lambda

    # Set timesteps
    overrides.setdefault("algorithm", {}).setdefault("training", {})["total_timesteps"] = args.timesteps
    overrides.setdefault("algorithm", {}).setdefault("training", {})["num_envs"] = args.num_envs
    overrides.setdefault("algorithm", {}).setdefault("training", {})["num_steps"] = args.num_steps
    overrides.setdefault("algorithm", {}).setdefault("training", {})["seed"] = args.seed

    # Merge overrides
    if overrides:
        merged_dict = config_manager._merge_dicts(config.to_dict(), overrides)
        config = SocialJaxConfig.from_dict(merged_dict)

    return config.to_dict()


def build_callbacks(args: argparse.Namespace) -> CallbackList:
    """Build callback list from arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        CallbackList with configured callbacks
    """
    callbacks = []

    # Checkpoint callback
    checkpoint_dir = os.path.join(
        args.checkpoint_dir,
        f"{args.algorithm}_{args.env}",
    )
    callbacks.append(
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=checkpoint_dir,
            name_prefix=args.algorithm,
            verbose=args.verbose,
        )
    )

    # WandB callback
    if args.wandb_project:
        wandb_name = args.wandb_name or f"{args.algorithm}_{args.env}_{int(time.time())}"
        callbacks.append(
            WandbCallback(
                project=args.wandb_project,
                name=wandb_name,
                entity=args.wandb_entity,
                verbose=args.verbose,
            )
        )

    return CallbackList(callbacks)


def print_training_info(args: argparse.Namespace, config: Dict[str, Any]):
    """Print training configuration information.

    Args:
        args: Parsed CLI arguments
        config: Training configuration
    """
    print("\n" + "=" * 60)
    print("SocialJax Training")
    print("=" * 60)
    print(f"Algorithm:     {args.algorithm}")
    print(f"Environment:   {args.env}")
    print(f"Timesteps:     {args.timesteps:,}")
    print(f"Seed:          {args.seed}")
    print(f"Num envs:      {args.num_envs}")
    print(f"Num steps:     {args.num_steps}")

    if args.lr:
        print(f"Learning rate: {args.lr}")
    if args.gamma:
        print(f"Gamma:         {args.gamma}")
    if args.gae_lambda:
        print(f"GAE lambda:    {args.gae_lambda}")

    print(f"\nCheckpoint dir: {args.checkpoint_dir}")
    print(f"Checkpoint freq: {args.checkpoint_freq:,} steps")

    if args.wandb_project:
        print(f"\nWandB project: {args.wandb_project}")
        print(f"WandB name:    {args.wandb_name or f'{args.algorithm}_{args.env}_{int(time.time())}'}")

    print("=" * 60 + "\n")


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def main():
    """Main training entry point."""
    global _interrupted

    args = parse_args()

    # Set up signal handler for graceful interrupt
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load configuration
    config = load_config(args)

    # Print training info
    if args.verbose >= 1:
        print_training_info(args, config)

    # Build callbacks
    callbacks = build_callbacks(args)

    # Create trainer
    if args.verbose >= 1:
        print("Creating trainer...")

    trainer = create_trainer(
        algorithm=args.algorithm,
        env=args.env,
        config=config,
        callbacks=callbacks.callbacks if hasattr(callbacks, 'callbacks') else list(callbacks),
        seed=args.seed,
    )

    if args.verbose >= 1:
        print("Starting training...\n")

    # Training loop
    start_time = time.time()
    try:
        state, metrics = trainer.train(
            total_timesteps=args.timesteps,
        )

        if not _interrupted:
            # Training completed successfully
            elapsed = time.time() - start_time

            print("\n" + "=" * 60)
            print("Training Complete!")
            print("=" * 60)
            print(f"Total timesteps: {state.timestep:,}")
            print(f"Total updates:   {state.update_step:,}")
            print(f"Total episodes:  {state.episode_count:,}")
            print(f"Elapsed time:    {format_time(elapsed)}")
            print(f"Steps/second:    {state.timestep / elapsed:.1f}")

            if metrics.get("episode_returns"):
                mean_return = sum(metrics["episode_returns"]) / len(metrics["episode_returns"])
                print(f"Mean episode return: {mean_return:.2f}")

            # Save final checkpoint
            final_path = os.path.join(
                args.checkpoint_dir,
                f"{args.algorithm}_{args.env}",
                f"{args.algorithm}_final",
            )
            trainer.save(final_path)
            print(f"\nFinal checkpoint saved to: {final_path}")
            print("=" * 60 + "\n")

    except KeyboardInterrupt:
        # This shouldn't happen due to signal handler, but just in case
        _interrupted = True

    finally:
        if _interrupted:
            # Save checkpoint on interrupt
            elapsed = time.time() - start_time
            interrupt_path = os.path.join(
                args.checkpoint_dir,
                f"{args.algorithm}_{args.env}",
                f"{args.algorithm}_interrupted",
            )

            print(f"Saving checkpoint to: {interrupt_path}")
            trainer.save(interrupt_path)

            print("\n" + "=" * 60)
            print("Training Interrupted")
            print("=" * 60)
            print(f"Timesteps completed: {trainer._state.timestep if hasattr(trainer, '_state') else 'unknown':,}")
            print(f"Elapsed time:        {format_time(elapsed)}")
            print(f"Checkpoint saved to: {interrupt_path}")
            print("=" * 60 + "\n")

            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
