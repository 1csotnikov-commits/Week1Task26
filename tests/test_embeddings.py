"""Тесты генерации эмбеддингов: Ollama (основной) и nomic (legacy).

Тесты nomic пропускаются, если модель недоступна (нет сети / не скачалась).
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.embeddings.ollama_provider import OllamaEmbeddingProvider  # noqa: E402


class TestOllamaEmbeddings(unittest.TestCase):
    def _provider(self, use_prefixes=True):
        return OllamaEmbeddingProvider(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434/v1",
            use_prefixes=use_prefixes,
            batch_size=2,
        )

    def test_encode_documents_adds_prefix_and_normalizes(self):
        captured = {}

        def fake_embed(base_url, model, inputs, timeout=300.0):
            captured["base_url"] = base_url
            captured["model"] = model
            captured["inputs"] = list(inputs)
            return [[3.0, 4.0] for _ in inputs]  # норма = 5, должна нормализоваться в 1

        provider = self._provider()
        with mock.patch("src.embeddings.ollama_provider.embed", side_effect=fake_embed):
            vectors = provider.encode_documents(["текст документа"])

        self.assertEqual(vectors.shape, (1, 2))
        self.assertEqual(captured["base_url"], "http://localhost:11434/v1")
        self.assertEqual(captured["model"], "nomic-embed-text")
        self.assertTrue(captured["inputs"][0].startswith("search_document:"))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0], atol=1e-6)
        self.assertEqual(provider.dim, 2)

    def test_encode_query_adds_search_query_prefix(self):
        captured = {}

        def fake_embed(base_url, model, inputs, timeout=300.0):
            captured["inputs"] = list(inputs)
            return [[1.0, 0.0] for _ in inputs]

        provider = self._provider()
        with mock.patch("src.embeddings.ollama_provider.embed", side_effect=fake_embed):
            vectors = provider.encode_query("Что такое ТС ПИоТ?")

        self.assertEqual(vectors.shape, (1, 2))
        self.assertEqual(captured["inputs"], ["search_query: Что такое ТС ПИоТ?"])

    def test_prefixes_disabled(self):
        captured = {}

        def fake_embed(base_url, model, inputs, timeout=300.0):
            captured["inputs"] = list(inputs)
            return [[1.0, 0.0] for _ in inputs]

        provider = self._provider(use_prefixes=False)
        with mock.patch("src.embeddings.ollama_provider.embed", side_effect=fake_embed):
            provider.encode_documents(["текст"])

        self.assertEqual(captured["inputs"], ["текст"])

    def test_batching_splits_requests(self):
        calls = []

        def fake_embed(base_url, model, inputs, timeout=300.0):
            calls.append(list(inputs))
            return [[1.0, 0.0, 0.0] for _ in inputs]

        provider = self._provider(use_prefixes=False)
        with mock.patch("src.embeddings.ollama_provider.embed", side_effect=fake_embed):
            vectors = provider.encode(["a", "b", "c"], batch_size=2)

        self.assertEqual(vectors.shape, (3, 3))
        self.assertEqual([len(c) for c in calls], [2, 1])


class TestNomicEmbeddings(unittest.TestCase):
    def test_nomic_provider_dim_and_normalization(self):
        try:
            import sentence_transformers  # noqa: F401
        except Exception:  # noqa: BLE001 - нет пакета или не грузится DLL torch
            self.skipTest("sentence-transformers недоступен")

        from src.embeddings.nomic_provider import NomicProvider

        provider = NomicProvider()
        try:
            vecs = provider.encode(["Привет, мир", "Второй русский текст"])
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"Модель не загрузилась: {exc}")

        self.assertEqual(vecs.shape, (2, 768))
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-4)


if __name__ == "__main__":
    unittest.main()

