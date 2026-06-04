from __future__ import annotations

from typing import Any, Callable, Type

from .base import BaseLLMBackend

_REGISTRY: dict[str, Type[BaseLLMBackend]] = {}


def register_backend(name: str) -> Callable[[Type[BaseLLMBackend]], Type[BaseLLMBackend]]:
    def deco(cls: Type[BaseLLMBackend]) -> Type[BaseLLMBackend]:
        if name in _REGISTRY:
            raise ValueError(f"Backend {name!r} already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def build_backend(name: str, config: dict[str, Any] | None = None) -> BaseLLMBackend:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown backend {name!r}. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](config=config)


def list_backends() -> list[str]:
    return sorted(_REGISTRY)
