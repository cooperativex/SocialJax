"""SocialJax algorithms module.

This module provides algorithm implementations and the registry system
for managing and discovering available algorithms.
"""

from socialjax.algorithms.registry import (
    register_algorithm,
    get_algorithm,
    list_algorithms,
    unregister_algorithm,
    is_algorithm_registered,
    clear_registry,
    AlgorithmAlreadyRegisteredError,
    AlgorithmNotFoundError,
)

__all__ = [
    # Registry functions
    "register_algorithm",
    "get_algorithm",
    "list_algorithms",
    "unregister_algorithm",
    "is_algorithm_registered",
    "clear_registry",
    # Exceptions
    "AlgorithmAlreadyRegisteredError",
    "AlgorithmNotFoundError",
]
