"""Абстракция генератора эмбеддингов."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

# Префиксы для модели nomic-embed-text-v2-moe.
# Модель различает типы задач: при индексации к тексту документа добавляется
# search_document:, при поиске к запросу — search_query:.
SEARCH_DOCUMENT_PREFIX = "search_document: "
SEARCH_QUERY_PREFIX = "search_query: "

# Маркер формата эмбеддингов, сохраняемый в индексе.
# Позволяет обнаружить старые индексы (созданные без префиксов) и предложить переиндексацию.
EMBEDDING_FORMAT_VERSION = "nomic-search-doc-v1"


class EmbeddingProvider(ABC):
    """Интерфейс генератора эмбеддингов.

    Реализации возвращают нормализованные векторы (L2-норма == 1),
    что позволяет использовать IndexFlatIP (inner product) как косинусную близость.
    """

    name: str = "base"
    dim: Optional[int] = None
    model_name: str = ""

    @abstractmethod
    def encode(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Вернуть матрицу эмбеддингов формы (len(texts), dim) без префиксов."""
        raise NotImplementedError

    def encode_documents(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Эмбеддинги документов (по умолчанию — с префиксом ``search_document:``)."""
        return self.encode([SEARCH_DOCUMENT_PREFIX + t for t in texts], batch_size)

    def encode_query(self, text: str, batch_size: Optional[int] = None) -> np.ndarray:
        """Эмбеддинг запроса (по умолчанию — с префиксом ``search_query:``)."""
        return self.encode([SEARCH_QUERY_PREFIX + text], batch_size)

