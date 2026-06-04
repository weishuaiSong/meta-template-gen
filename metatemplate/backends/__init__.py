"""LLM backends. Each backend imports its heavy SDK (openai / anthropic / vllm)
lazily inside ``__init__``, so the backend always *registers* (and lists), and
only raises a helpful ImportError when you actually build one without its dep
installed. Importing this package never requires any of those SDKs."""
from __future__ import annotations

from .base import BaseLLMBackend, GenRequest
from .registry import build_backend, list_backends, register_backend

# Register all backends. The try/except is a defensive guard for module-level
# import errors only; missing SDKs surface at build_backend() time, not here.
for _mod in ("openai_compat", "anthropic_backend", "vllm_offline", "hf_local"):
    try:
        __import__(f"{__name__}.{_mod}", fromlist=["*"])
    except ImportError:
        pass

__all__ = ["BaseLLMBackend", "GenRequest", "build_backend", "list_backends", "register_backend"]
