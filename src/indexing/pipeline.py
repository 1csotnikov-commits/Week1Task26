"""Пайплайн построения индекса: загрузка -> чанкинг -> эмбеддинги -> хранение."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple

from src.chunking import make_chunker
from src.config import Config
from src.embeddings import make_embedder
from src.embeddings.base import EMBEDDING_FORMAT_VERSION, EmbeddingProvider
from src.indexing.faiss_sqlite_store import FAISSSQLiteStore
from src.ingestion.text_loader import TextDocumentLoader
from src.models import Chunk, Document


class BuildResult:
    """Результат построения индекса."""

    def __init__(self, documents: List[Document], chunks: List[Chunk], elapsed: float, index_name: str, strategy: str):
        self.documents = documents
        self.chunks = chunks
        self.elapsed = elapsed
        self.index_name = index_name
        self.strategy = strategy


def build_index(
    documents_path: str | Path,
    strategy: str,
    index_name: str,
    config: Config,
    provider: Optional[EmbeddingProvider] = None,
) -> BuildResult:
    """Построить индекс из документов по выбранной стратегии чанкинга."""
    started = time.time()

    loader = TextDocumentLoader(extensions=config.document_extensions)
    documents = loader.load(documents_path)
    if not documents:
        raise ValueError(f"В '{documents_path}' не найдено текстовых документов с расширениями {config.document_extensions}.")

    chunker = make_chunker(strategy, config)
    chunks: List[Chunk] = []
    for doc in documents:
        chunks.extend(chunker.chunk(doc))
    if not chunks:
        raise ValueError("После чанкинга не получено ни одного чанка. Проверьте содержимое документов.")

    if provider is None:
        provider = make_embedder(config)

    texts = [c.text for c in chunks]
    print(f"Генерация эмбеддингов для {len(texts)} чанков (провайдер: {provider.name})...")
    # Провайдер сам добавляет нужный префикс (для nomic — search_document:).
    embeddings = provider.encode_documents(texts)

    store = FAISSSQLiteStore(config.index_path)
    store.begin(
        index_name=index_name,
        strategy=strategy,
        model=getattr(provider, "model_name", "") or config.embedding_model,
        provider=getattr(provider, "name", "") or config.embedding_provider,
        dim=int(embeddings.shape[1]),
        embedding_format=EMBEDDING_FORMAT_VERSION,
    )
    for doc in documents:
        store.add_document(doc)
    for chunk, emb in zip(chunks, embeddings):
        store.add(chunk, emb)
    store.finalize(num_documents=len(documents))

    elapsed = time.time() - started
    return BuildResult(documents, chunks, elapsed, index_name, strategy)
