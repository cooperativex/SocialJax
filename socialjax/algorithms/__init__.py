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

# Import algorithm implementations to register them
from socialjax.algorithms import ippo
from socialjax.algorithms import mappo
from socialjax.algorithms import vdn
from socialjax.algorithms import svo

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
    # Algorithm modules
    "ippo",
    "mappo",
    "vdn",
    "svo",
]
