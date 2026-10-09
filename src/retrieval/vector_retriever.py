"""Векторный поиск по FAISS-индексу с метаданными из SQLite."""

from __future__ import annotations

from typing import List, Optional

from src.embeddings.base import EMBEDDING_FORMAT_VERSION, EmbeddingProvider
from src.indexing.faiss_sqlite_store import FAISSSQLiteStore
from src.retrieval.base import Retriever
from src.retrieval.models import RetrievedChunk


class VectorRetriever(Retriever):
    """Поиск: эмбеддинг запроса (с префиксом search_query:) -> FAISS -> метаданные."""

    name = "vector"

    def __init__(self, store: FAISSSQLiteStore, embedder: EmbeddingProvider, index_name: Optional[str] = None):
        self.store = store
        self.embedder = embedder
        self.index_name = index_name

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        if self.index_name is None:
            raise ValueError("Не указано имя индекса для поиска (передайте index_name).")

        self.store.load(self.index_name)
        self._check_embedding_format()
        self._check_embedding_model()

        # Запрос эмбеддится тем же провайдером, что и документы (для nomic — с префиксом search_query:).
        query_embedding = self.embedder.encode_query(query)[0]

        hits = self.store.search(query_embedding, top_k)
        return [
            RetrievedChunk(chunk=h.chunk, score=h.score, rank=i + 1)
            for i, h in enumerate(hits)
        ]

    def _check_embedding_format(self) -> None:
        fmt = self.store.embedding_format
        if fmt == EMBEDDING_FORMAT_VERSION:
            return
        print(
            f"ПРЕДУПРЕЖДЕНИЕ: индекс '{self.index_name}' создан в другом формате эмбеддингов "
            f"(ожидается '{EMBEDDING_FORMAT_VERSION}', получен '{fmt or 'старый формат без префиксов'}'). "
            "Результаты поиска могут быть неточными. Переиндексируйте документы: "
            f"/index build <path> --name {self.index_name}"
        )

    def _check_embedding_model(self) -> None:
        index_model = getattr(self.store, "index_model", "") or ""
        embedder_model = getattr(self.embedder, "model_name", "") or ""
        if index_model and embedder_model and index_model != embedder_model:
            print(
                f"ПРЕДУПРЕЖДЕНИЕ: индекс '{self.index_name}' построен моделью эмбеддингов "
                f"'{index_model}', а поиск идёт моделью '{embedder_model}'. Результаты могут быть "
                f"неточными. Переиндексируйте документы: /index build <path> --name {self.index_name}"
            )
