from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenRequest:
    system: str
    user: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseLLMBackend(ABC):
    """Text-in / text-out LLM backend.

    Implementations are batch-friendly: ``complete`` takes a list of requests and
    returns one string per request, in order. Note that templates are *format*,
    not answers — a teacher LLM here only authors the question wrapper and never
    touches label generation.
    """

    name: str = "base"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})

    @abstractmethod
    def complete(self, requests: list[GenRequest]) -> list[str]:
        ...

    def shutdown(self) -> None:
        return None
