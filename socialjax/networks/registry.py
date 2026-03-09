"""Network registry for SocialJax."""
from typing import Dict, Type, List, Optional, Any
from flax import linen as nn

_NETWORK_REGISTRY: Dict[str, Type[nn.Module]] = {}

class NetworkAlreadyRegisteredError(Exception):
    pass

class NetworkNotFoundError(Exception):
    pass

def register_network(name: str):
    def decorator(network_class: Type[nn.Module]) -> Type[nn.Module]:
        if name in _NETWORK_REGISTRY:
            raise NetworkAlreadyRegisteredError(
                f"Network '{name}' is already registered.")
        _NETWORK_REGISTRY[name] = network_class
        return network_class
    return decorator

def get_network_class(name: str) -> Type[nn.Module]:
    if name not in _NETWORK_REGISTRY:
        raise NetworkNotFoundError(
            f"Network '{name}' not found. Available: {list_networks()}")
    return _NETWORK_REGISTRY[name]

def list_networks() -> List[str]:
    return sorted(_NETWORK_REGISTRY.keys())

def unregister_network(name: str) -> Optional[Type[nn.Module]]:
    return _NETWORK_REGISTRY.pop(name, None)

def is_network_registered(name: str) -> bool:
    return name in _NETWORK_REGISTRY

def clear_registry() -> None:
    _NETWORK_REGISTRY.clear()
