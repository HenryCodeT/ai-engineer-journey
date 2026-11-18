"""Utilidades de LLM para ejercicios y tests del AI Engineer Journey."""
from .fake_llm import CATALOGO, FakeLLM, LLMError, Respuesta, contar_tokens
from .stub_llm import StubLLM, error

__all__ = [
    "CATALOGO",
    "FakeLLM",
    "LLMError",
    "Respuesta",
    "contar_tokens",
    "StubLLM",
    "error",
]

try:
    from .claude_llm import ClaudeLLM, crear_llm

    __all__.extend(["ClaudeLLM", "crear_llm"])
except ImportError:
    pass
