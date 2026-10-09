"""Хранилище индексов: FAISS (векторы) + SQLite (метаданные и текст чанков).

Если faiss-cpu не устанавливается на Windows, автоматически используется
fallback на numpy + перебор (для 20–30 страниц это приемлемо по скорости).
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from src.indexing.base import IndexStore
from src.models import Chunk, ChunkWithScore, Document

try:
    import faiss  # type: ignore

    HAS_FAISS = True
except ImportError:  # pragma: no cover - зависит от окружения
    faiss = None
    HAS_FAISS = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class FAISSSQLiteStore(IndexStore):
    """Реализация IndexStore поверх FAISS (IndexFlatIP) и SQLite."""

    def __init__(self, index_root: str | Path):
        self.index_root = Path(index_root)
        self._db: Optional[sqlite3.Connection] = None
        self._index_name: Optional[str] = None
        self._faiss_index = None
        self._vectors: Optional[np.ndarray] = None
        self._chunk_ids: List[str] = []
        self._strategy = ""
        self._model = ""
        self._provider = ""
        self._dim = 0
        self._num_documents = 0
        self._started = 0.0
        self._embedding_format = ""

    # ------------------------------------------------------------------ #
    # Пути
    # ------------------------------------------------------------------ #
    def _index_dir(self, name: str) -> Path:
        return self.index_root / name

    def _faiss_path(self, name: str) -> Path:
        return self._index_dir(name) / "faiss.index"

    def _vectors_path(self, name: str) -> Path:
        return self._index_dir(name) / "vectors.npy"

    def _db_path(self, name: str) -> Path:
        return self._index_dir(name) / "metadata.db"

    # ------------------------------------------------------------------ #
    # Построение индекса
    # ------------------------------------------------------------------ #
    def begin(self, index_name: str, strategy: str, model: str, provider: str, dim: int, embedding_format: str = "") -> None:
        if HAS_FAISS:
            print("Используется FAISS IndexFlatIP (inner product = косинусная близость).")
        else:
            print(
                "ПРЕДУПРЕЖДЕНИЕ: библиотека faiss не установлена. "
                "Используется fallback на numpy + перебор (для 20–30 страниц скорость приемлема)."
            )

        self._index_name = index_name
        self._strategy = strategy
        self._model = model
        self._provider = provider
        self._dim = dim
        self._embedding_format = embedding_format
        self._num_documents = 0
        self._chunk_ids = []
        self._started = time.time()

        # Повторная сборка с тем же именем перезаписывает старый индекс целиком.
        index_dir = self._index_dir(index_name)
        if index_dir.exists():
            print(f"Индекс '{index_name}' уже существует — перезаписываю (удаляю старую версию).")
            shutil.rmtree(index_dir)

        index_dir.mkdir(parents=True, exist_ok=True)
        self._open_db()
        self._create_tables()

        self._faiss_index = faiss.IndexFlatIP(dim) if HAS_FAISS else None
        self._vectors = np.empty((0, dim), dtype=np.float32)

    def add_document(self, document: Document) -> None:
        self._exec(
            "INSERT INTO documents (source, title, format, char_count, token_count) VALUES (?, ?, ?, ?, ?)",
            (document.source, document.title, document.format, document.char_count, document.token_count),
        )
        self._num_documents += 1

    def add(self, chunk: Chunk, embedding: np.ndarray) -> None:
        self._insert_chunk(chunk)
        self._chunk_ids.append(chunk.chunk_id)

        vec = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if HAS_FAISS:
            self._faiss_index.add(vec)
        else:
            self._vectors = np.vstack([self._vectors, vec])

    def finalize(self, num_documents: int) -> None:
        if not self._index_name:
            raise RuntimeError("finalize вызван до begin()")

        self._exec(
            "INSERT OR REPLACE INTO indices (name, strategy, model, provider, dim, store_type, num_documents, num_chunks, indexing_time_seconds, created_at, embedding_format) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self._index_name,
                self._strategy,
                self._model,
                self._provider,
                self._dim,
                "faiss" if HAS_FAISS else "numpy",
                num_documents,
                len(self._chunk_ids),
                round(time.time() - self._started, 4),
                _now_iso(),
                self._embedding_format,
            ),
        )
        self._db.commit()

        if HAS_FAISS:
            faiss.write_index(self._faiss_index, str(self._faiss_path(self._index_name)))
        else:
            np.save(self._vectors_path(self._index_name), self._vectors)

        self._db.commit()
        print(
            f"Индекс '{self._index_name}' сохранён: {self._index_dir(self._index_name)} "
            f"({len(self._chunk_ids)} чанков, хранилище: {'faiss' if HAS_FAISS else 'numpy'})."
        )
        self._close_db()

    # ------------------------------------------------------------------ #
    # SQLite helpers
    # ------------------------------------------------------------------ #
    def _open_db(self) -> None:
        self._db = sqlite3.connect(str(self._db_path(self._index_name)))

    def _close_db(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    def _exec(self, sql: str, params: tuple = ()) -> None:
        if self._db is None:
            raise RuntimeError("База данных не открыта")
        self._db.execute(sql, params)

    def _create_tables(self) -> None:
        self._exec(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                title TEXT,
                format TEXT,
                char_count INTEGER,
                token_count INTEGER
            )
            """
        )
        self._exec(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT NOT NULL UNIQUE,
                source TEXT,
                title TEXT,
                section TEXT,
                strategy TEXT,
                text TEXT,
                start_char INTEGER,
                end_char INTEGER,
                char_count INTEGER,
                token_count INTEGER,
                metadata TEXT
            )
            """
        )
        self._exec(
            """
            CREATE TABLE IF NOT EXISTS indices (
                name TEXT PRIMARY KEY,
                strategy TEXT,
                model TEXT,
                provider TEXT,
                dim INTEGER,
                store_type TEXT,
                num_documents INTEGER,
                num_chunks INTEGER,
                indexing_time_seconds REAL,
                created_at TEXT,
                embedding_format TEXT
            )
            """
        )
        self._db.commit()

    def _insert_chunk(self, chunk: Chunk) -> None:
        self._exec(
            "INSERT INTO chunks (chunk_id, source, title, section, strategy, text, start_char, end_char, char_count, token_count, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                chunk.chunk_id,
                chunk.source,
                chunk.title,
                chunk.section,
                chunk.strategy,
                chunk.text,
                chunk.start_char,
                chunk.end_char,
                chunk.char_count,
                chunk.token_count,
                json.dumps(chunk.metadata, ensure_ascii=False),
            ),
        )

    @staticmethod
    def _row_to_chunk(row: tuple) -> Chunk:
        (chunk_id, source, title, section, strategy, text, start_char, end_char, char_count, token_count, metadata) = row
        try:
            meta = json.loads(metadata) if metadata else {}
        except json.JSONDecodeError:
            meta = {}
        return Chunk(
            chunk_id=chunk_id,
            source=source or "",
            title=title or "",
            section=section or "",
            strategy=strategy or "",
            text=text or "",
            start_char=start_char or 0,
            end_char=end_char or 0,
            char_count=char_count or len(text or ""),
            token_count=token_count or 0,
            metadata=meta,
        )

    _CHUNK_COLUMNS = "chunk_id, source, title, section, strategy, text, start_char, end_char, char_count, token_count, metadata"

    def _fetchall(self, sql: str, params: tuple = ()) -> List[tuple]:
        conn = self._connect()
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    def _fetchone(self, sql: str, params: tuple = ()) -> Optional[tuple]:
        conn = self._connect()
        try:
            return conn.execute(sql, params).fetchone()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path(self._index_name)))

    # ------------------------------------------------------------------ #
    # Чтение и поиск
    # ------------------------------------------------------------------ #
    def load(self, index_name: str) -> None:
        if not self._index_dir(index_name).exists():
            raise ValueError(f"Индекс '{index_name}' не найден в {self.index_root}")

        if self._db is not None:
            self._close_db()

        self._index_name = index_name

        store_type = self._get_store_type(index_name)
        if store_type == "faiss":
            if not HAS_FAISS:
                raise RuntimeError(
                    f"Индекс '{index_name}' создан с FAISS, но faiss не установлен в текущем окружении."
                )
            self._faiss_index = faiss.read_index(str(self._faiss_path(index_name)))
            self._vectors = None
        else:
            self._faiss_index = None
            self._vectors = np.load(self._vectors_path(index_name))

        self._chunk_ids = self._ordered_chunk_ids()
        row = self._fetchone("SELECT strategy, model, provider, dim FROM indices WHERE name = ?", (index_name,))
        if row:
            self._strategy, self._model, self._provider, self._dim = row
        self._embedding_format = self._read_embedding_format(index_name)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[ChunkWithScore]:
        if self._index_name is None:
            raise RuntimeError("Индекс не загружен. Вызовите load() или finalize().")
        if not self._chunk_ids:
            return []

        q = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if self._faiss_index is not None:
            k = min(top_k, self._faiss_index.ntotal)
            scores, ids = self._faiss_index.search(q, k)
            scores = scores.flatten()
            ids = ids.flatten()
        else:
            sims = self._vectors @ q[0]
            ids = np.argsort(-sims)[:top_k]
            scores = sims[ids]

        results: List[ChunkWithScore] = []
        for pos, score in zip(ids, scores):
            if pos < 0 or int(pos) >= len(self._chunk_ids):
                continue
            chunk_id = self._chunk_ids[int(pos)]
            chunk = self._get_chunk_by_id(chunk_id)
            if chunk is not None:
                results.append(ChunkWithScore(chunk=chunk, score=float(score)))
        return results

    def _get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        row = self._fetchone(f"SELECT {self._CHUNK_COLUMNS} FROM chunks WHERE chunk_id = ?", (chunk_id,))
        return self._row_to_chunk(row) if row else None

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self._get_chunk_by_id(chunk_id)

    def get_embedding(self, chunk_id: str) -> Optional[np.ndarray]:
        if chunk_id not in self._chunk_ids:
            return None
        pos = self._chunk_ids.index(chunk_id)
        if self._faiss_index is not None:
            return np.asarray(self._faiss_index.reconstruct(pos), dtype=np.float32)
        return self._vectors[pos].copy()

    def get_documents(self) -> List[Document]:
        """Вернуть документы из таблицы documents (без текста — только метаданные)."""
        rows = self._fetchall("SELECT source, title, format, char_count, token_count FROM documents ORDER BY id")
        return [
            Document(
                source=r[0],
                title=r[1] or "",
                text="",
                format=r[2] or "",
                char_count=r[3] or 0,
                token_count=r[4] or 0,
            )
            for r in rows
        ]

    def list_chunks(self, source: Optional[str] = None, section: Optional[str] = None, limit: Optional[int] = None) -> List[Chunk]:
        sql = f"SELECT {self._CHUNK_COLUMNS} FROM chunks WHERE 1=1"
        params: List[Any] = []
        if source:
            sql += " AND source = ?"
            params.append(source)
        if section:
            sql += " AND section = ?"
            params.append(section)
        sql += " ORDER BY id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return [self._row_to_chunk(r) for r in self._fetchall(sql, tuple(params))]

    # ------------------------------------------------------------------ #
    # Метаданные индексов
    # ------------------------------------------------------------------ #
    def _get_store_type(self, index_name: str) -> str:
        row = self._fetchone("SELECT store_type FROM indices WHERE name = ?", (index_name,))
        if row and row[0]:
            return row[0]
        if self._faiss_path(index_name).exists():
            return "faiss"
        return "numpy"

    def _ordered_chunk_ids(self) -> List[str]:
        rows = self._fetchall("SELECT chunk_id FROM chunks ORDER BY id")
        return [r[0] for r in rows]

    @property
    def embedding_format(self) -> str:
        """Формат эмбеддингов загруженного/построенного индекса ('' для старых индексов)."""
        return self._embedding_format

    @property
    def index_model(self) -> str:
        """Модель эмбеддингов загруженного индекса ('' если неизвестна)."""
        return self._model

    @property
    def index_provider(self) -> str:
        """Провайдер эмбеддингов загруженного индекса ('' если неизвестен)."""
        return self._provider

    def _read_embedding_format(self, index_name: str) -> str:
        try:
            row = self._fetchone("SELECT embedding_format FROM indices WHERE name = ?", (index_name,))
        except sqlite3.OperationalError:
            return ""
        return (row[0] if row and row[0] else "")

    def info(self, index_name: str) -> dict:
        db_path = self._db_path(index_name)
        if not db_path.exists():
            raise ValueError(f"Индекс '{index_name}' не найден в {self.index_root}")

        conn = sqlite3.connect(str(db_path))
        try:
            try:
                row = conn.execute(
                    "SELECT name, strategy, model, provider, dim, store_type, num_documents, num_chunks, indexing_time_seconds, created_at, embedding_format "
                    "FROM indices WHERE name = ?",
                    (index_name,),
                ).fetchone()
                has_format = True
            except sqlite3.OperationalError:
                row = conn.execute(
                    "SELECT name, strategy, model, provider, dim, store_type, num_documents, num_chunks, indexing_time_seconds, created_at "
                    "FROM indices WHERE name = ?",
                    (index_name,),
                ).fetchone()
                has_format = False
            num_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            num_documents = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        finally:
            conn.close()

        info = {
            "name": index_name,
            "path": str(self._index_dir(index_name)),
            "store_type": "numpy",
            "strategy": "",
            "model": "",
            "provider": "",
            "dim": 0,
            "num_documents": num_documents,
            "num_chunks": num_chunks,
            "indexing_time_seconds": 0.0,
            "created_at": "",
            "embedding_format": "",
        }
        if row:
            info.update({
                "name": row[0],
                "strategy": row[1] or "",
                "model": row[2] or "",
                "provider": row[3] or "",
                "dim": row[4] or 0,
                "store_type": row[5] or "numpy",
                "num_documents": row[6] or num_documents,
                "num_chunks": row[7] or num_chunks,
                "indexing_time_seconds": row[8] or 0.0,
                "created_at": row[9] or "",
                "embedding_format": (row[10] if has_format and row[10] else ""),
            })
        return info

    def list_indices(self) -> List[dict]:
        if not self.index_root.exists():
            return []
        result = []
        for d in sorted(self.index_root.iterdir()):
            if d.is_dir() and (d / "metadata.db").exists():
                result.append(self.info(d.name))
        return result

    def delete(self, index_name: str) -> None:
        if self._index_name == index_name and self._db is not None:
            self._close_db()
            self._index_name = None
            self._faiss_index = None
            self._vectors = None
            self._chunk_ids = []
        d = self._index_dir(index_name)
        if d.exists():
            shutil.rmtree(d)
        else:
            raise ValueError(f"Индекс '{index_name}' не найден")
