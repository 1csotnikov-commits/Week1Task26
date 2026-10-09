"""LLM-провайдеры.

Основной (по умолчанию) провайдер — ``ollama`` (локальные модели Ollama).
Провайдер ``deepseek`` оставлен как необязательная альтернатива.
"""

from __future__ import annotations

from typing import Any, Optional

from src.llm.base import LLMProvider, LLMResponse
from src.llm.deepseek_provider import DeepSeekProvider
from src.llm.ollama_provider import DEFAULT_MODEL as OLLAMA_DEFAULT_MODEL
from src.llm.ollama_provider import OllamaProvider

LLM_PROVIDERS = ("ollama", "deepseek")


def _cfg(config: Any, dotted: str, default: Any) -> Any:
    """Прочитать значение из конфига (duck-typing: нужен метод ``get``)."""
    if config is None:
        return default
    return config.get(dotted, default)


def make_llm_provider(config: Any = None, provider: Optional[str] = None) -> LLMProvider:
    """Создать LLM-провайдер по имени из конфига (по умолчанию — ollama)."""
    provider = (provider or _cfg(config, "llm.provider", "ollama") or "ollama").lower()

    if provider == "ollama":
        host = _cfg(config, "llm.host", "") or _cfg(config, "ollama.host", "http://localhost:11434")
        return OllamaProvider(
            model=_cfg(config, "llm.model", OLLAMA_DEFAULT_MODEL) or OLLAMA_DEFAULT_MODEL,
            host=host,
            temperature=float(_cfg(config, "llm.temperature", 0.3)),
            max_tokens=int(_cfg(config, "llm.max_tokens", 1024)),
            timeout=float(_cfg(config, "llm.timeout", 300.0)),
            num_ctx=_cfg(config, "llm.num_ctx", 8192),
            keep_alive=_cfg(config, "llm.keep_alive", "5m"),
            top_p=_cfg(config, "llm.top_p", None),
        )

    if provider == "deepseek":
        return DeepSeekProvider(
            model=_cfg(config, "llm.model", "deepseek-chat"),
            base_url=_cfg(config, "llm.base_url", "https://api.deepseek.com"),
            temperature=float(_cfg(config, "llm.temperature", 0.3)),
            max_tokens=int(_cfg(config, "llm.max_tokens", 1024)),
            timeout=float(_cfg(config, "llm.timeout", 60.0)),
        )

    raise ValueError(
        f"Неизвестный LLM-провайдер: {provider!r}. Доступно: {', '.join(LLM_PROVIDERS)}."
    )


__all__ = [
    "LLMProvider",
    "LLMResponse",
    "OllamaProvider",
    "DeepSeekProvider",
    "make_llm_provider",
    "LLM_PROVIDERS",
]

