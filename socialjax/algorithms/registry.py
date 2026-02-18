"""Algorithm registry for SocialJax.

This module provides a registry system for algorithms, enabling dynamic
discovery and loading of algorithm implementations. Algorithms can be
registered using the @register_algorithm decorator and retrieved using
get_algorithm().

Example usage:
    @register_algorithm('ippo')
    class IPPOAlgorithm(BaseAlgorithm):
        ...

    # Later, retrieve the algorithm
    algo_class = get_algorithm('ippo')
    algo = algo_class(config)
"""

from typing import Dict, Type, List, Optional, Any
from socialjax.core.base_algorithm import BaseAlgorithm


# Private registry dictionary
_ALGORITHM_REGISTRY: Dict[str, Type[BaseAlgorithm]] = {}


class AlgorithmAlreadyRegisteredError(Exception):
    """Raised when attempting to register an algorithm name that already exists."""
    pass


class AlgorithmNotFoundError(Exception):
    """Raised when attempting to get an algorithm that hasn't been registered."""
    pass


def register_algorithm(name: str):
    """Decorator to register an algorithm class with the registry.

    Args:
        name: The name to register the algorithm under. This name will be used
              to retrieve the algorithm via get_algorithm().

    Returns:
        A decorator function that registers the algorithm class.

    Raises:
        AlgorithmAlreadyRegisteredError: If an algorithm with this name is
                                         already registered.

    Example:
        @register_algorithm('ippo')
        class IPPOAlgorithm(BaseAlgorithm):
            ...
    """
    def decorator(algo_class: Type[BaseAlgorithm]) -> Type[BaseAlgorithm]:
        if name in _ALGORITHM_REGISTRY:
            raise AlgorithmAlreadyRegisteredError(
                f"Algorithm '{name}' is already registered. "
                f"Existing: {_ALGORITHM_REGISTRY[name].__name__}, "
                f"Attempted: {algo_class.__name__}. "
                f"Use a different name or unregister the existing algorithm first."
            )
        _ALGORITHM_REGISTRY[name] = algo_class
        return algo_class
    return decorator


def get_algorithm(name: str) -> Type[BaseAlgorithm]:
    """Retrieve an algorithm class from the registry.

    Args:
        name: The name of the algorithm to retrieve.

    Returns:
        The registered algorithm class.

    Raises:
        AlgorithmNotFoundError: If no algorithm with the given name exists.

    Example:
        algo_class = get_algorithm('ippo')
        algo = algo_class(config)
    """
    if name not in _ALGORITHM_REGISTRY:
        available = list_algorithms()
        raise AlgorithmNotFoundError(
            f"Algorithm '{name}' not found. "
            f"Available algorithms: {available}. "
            f"Make sure the algorithm is registered using @register_algorithm('{name}')."
        )
    return _ALGORITHM_REGISTRY[name]


def list_algorithms() -> List[str]:
    """List all registered algorithm names.

    Returns:
        A sorted list of all registered algorithm names.

    Example:
        names = list_algorithms()
        print(names)  # ['ippo', 'mappo', 'vdn', ...]
    """
    return sorted(_ALGORITHM_REGISTRY.keys())


def unregister_algorithm(name: str) -> Optional[Type[BaseAlgorithm]]:
    """Remove an algorithm from the registry.

    This is primarily useful for testing and advanced use cases.

    Args:
        name: The name of the algorithm to unregister.

    Returns:
        The unregistered algorithm class, or None if the name wasn't registered.

    Example:
        algo_class = unregister_algorithm('ippo')
    """
    return _ALGORITHM_REGISTRY.pop(name, None)


def is_algorithm_registered(name: str) -> bool:
    """Check if an algorithm is registered.

    Args:
        name: The name of the algorithm to check.

    Returns:
        True if the algorithm is registered, False otherwise.

    Example:
        if is_algorithm_registered('ippo'):
            algo_class = get_algorithm('ippo')
    """
    return name in _ALGORITHM_REGISTRY


def clear_registry() -> None:
    """Clear all registered algorithms.

    This is primarily useful for testing.
    """
    _ALGORITHM_REGISTRY.clear()
