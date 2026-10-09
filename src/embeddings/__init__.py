"""Генерация эмбеддингов.

Основной (по умолчанию) провайдер — ``ollama`` (локальная embedding-модель Ollama).
Провайдер ``nomic`` (локальная HF-модель через sentence-transformers) оставлен
как необязательная альтернатива.
"""

from __future__ import annotations

from typing import Any, Optional

from src.embeddings.base import EMBEDDING_FORMAT_VERSION, EmbeddingProvider
from src.embeddings.nomic_provider import DEFAULT_MODEL as NOMIC_DEFAULT_MODEL
from src.embeddings.nomic_provider import NomicProvider
from src.embeddings.ollama_provider import DEFAULT_BASE_URL as OLLAMA_DEFAULT_EMBED_URL
from src.embeddings.ollama_provider import DEFAULT_MODEL as OLLAMA_DEFAULT_EMBED_MODEL
from src.embeddings.ollama_provider import OllamaEmbeddingProvider

EMBEDDING_PROVIDERS = ("ollama", "nomic")


def _cfg(config: Any, dotted: str, default: Any) -> Any:
    if config is None:
        return default
    return config.get(dotted, default)


def make_embedder(
    config: Any = None,
    model_override: Optional[str] = None,
    provider_override: Optional[str] = None,
) -> EmbeddingProvider:
    """Создать генератор эмбеддингов по имени из конфига (по умолчанию — ollama).

    ``model_override`` позволяет взять модель из метаданных индекса, чтобы поиск
    использовал ту же модель, которой индекс был построен.
    """
    provider = (provider_override or _cfg(config, "embedding.provider", "ollama") or "ollama").lower()

    if provider == "ollama":
        host = _cfg(config, "ollama.host", "http://localhost:11434")
        base_url = _cfg(config, "embedding.base_url", "") or f"{str(host).rstrip('/')}/v1"
        return OllamaEmbeddingProvider(
            model_name=model_override or _cfg(config, "embedding.model", OLLAMA_DEFAULT_EMBED_MODEL) or OLLAMA_DEFAULT_EMBED_MODEL,
            base_url=base_url,
            batch_size=int(_cfg(config, "embedding.batch_size", 16)),
            timeout=float(_cfg(config, "embedding.timeout", 300.0)),
            normalize=bool(_cfg(config, "embedding.normalize", True)),
            use_prefixes=bool(_cfg(config, "embedding.use_prefixes", True)),
        )

    if provider == "nomic":
        return NomicProvider(
            model_name=model_override or _cfg(config, "embedding.model", NOMIC_DEFAULT_MODEL) or NOMIC_DEFAULT_MODEL,
            device=_cfg(config, "embedding.device", "cpu"),
            batch_size=int(_cfg(config, "embedding.batch_size", 16)),
            trust_remote_code=bool(_cfg(config, "embedding.trust_remote_code", True)),
        )

    raise ValueError(
        f"Неизвестный провайдер эмбеддингов: {provider!r}. Доступно: {', '.join(EMBEDDING_PROVIDERS)}."
    )


__all__ = [
    "EMBEDDING_FORMAT_VERSION",
    "EmbeddingProvider",
    "OllamaEmbeddingProvider",
    "NomicProvider",
    "make_embedder",
    "EMBEDDING_PROVIDERS",
    "OLLAMA_DEFAULT_EMBED_MODEL",
    "OLLAMA_DEFAULT_EMBED_URL",
]

