"""Реализация EmbeddingProvider на базе локальных моделей Ollama.

Эмбеддинги берутся через OpenAI-совместимый эндпоинт ``/v1/embeddings``
(нативный ``/api/embed`` в некоторых сборках Ollama отключён — тогда сервер
просит запускать его с ``--embeddings``; OpenAI-совместимый эндпоинт работает всегда).

Модель выбирается из установленных в Ollama (см. команду ``/models``).
Рекомендуемая embedding-модель — ``nomic-embed-text`` (768 измерений, русский язык).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.embeddings.base import SEARCH_DOCUMENT_PREFIX, SEARCH_QUERY_PREFIX, EmbeddingProvider
from src.ollama.client import OllamaError, embed

DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_BASE_URL = "http://localhost:11434/v1"


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Эмбеддинги через локальный сервер Ollama."""

    name = "ollama"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        batch_size: int = 16,
        timeout: float = 300.0,
        normalize: bool = True,
        use_prefixes: bool = True,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.batch_size = batch_size
        self.timeout = timeout
        self.normalize = normalize
        self.use_prefixes = use_prefixes
        self.dim: Optional[int] = None

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        try:
            vectors = embed(self.base_url, self.model_name, texts, timeout=self.timeout)
        except OllamaError as exc:
            raise RuntimeError(f"Ошибка эмбеддингов Ollama: {exc}") from exc
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != len(texts):
            raise RuntimeError(
                f"Некорректная форма эмбеддингов от Ollama: {arr.shape}, ожидалось ({len(texts)}, dim)."
            )
        if self.normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        self.dim = int(arr.shape[1])
        return arr

    def encode(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        if not texts:
            dim = self.dim or 0
            return np.empty((0, dim), dtype=np.float32)

        bs = batch_size or self.batch_size
        batches = [self._embed_batch(list(texts[i : i + bs])) for i in range(0, len(texts), bs)]
        return np.vstack(batches)

    def encode_documents(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        prepared = [SEARCH_DOCUMENT_PREFIX + t for t in texts] if self.use_prefixes else list(texts)
        return self.encode(prepared, batch_size)

    def encode_query(self, text: str, batch_size: Optional[int] = None) -> np.ndarray:
        prepared = SEARCH_QUERY_PREFIX + text if self.use_prefixes else text
        return self.encode([prepared], batch_size)
