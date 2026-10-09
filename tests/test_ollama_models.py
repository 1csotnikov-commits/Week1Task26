"""Тесты клиента Ollama и команды /models (выбор локальных моделей)."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli import main  # noqa: E402
from src.cli.commands import build_help  # noqa: E402
from src.cli.main import App, handle_command  # noqa: E402
from src.config import Config  # noqa: E402
from src.ollama.client import list_models  # noqa: E402

TAGS_RESPONSE = {
    "models": [
        {
            "name": "gemma3:12b-it-qat",
            "size": 8928675346,
            "details": {"parameter_size": "12.2B", "quantization_level": "Q4_0", "family": "gemma3"},
            "capabilities": ["completion", "vision"],
        },
        {
            "name": "nomic-embed-text:latest",
            "size": 274300000,
            "details": {"parameter_size": "137M", "quantization_level": "F16", "family": "nomic-bert"},
            "capabilities": ["embedding"],
        },
    ]
}


def _fake_models():
    """Вернуть список моделей в формате клиента, не обращаясь к серверу."""
    with mock.patch("src.ollama.client._request_json", return_value=TAGS_RESPONSE):
        return list_models("http://localhost:11434")


class TestOllamaClient(unittest.TestCase):
    def test_list_models_parsing(self):
        models = _fake_models()
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0]["name"], "gemma3:12b-it-qat")
        self.assertEqual(models[0]["quantization"], "Q4_0")
        self.assertEqual(models[1]["capabilities"], ["embedding"])
        self.assertGreater(models[0]["size"], 0)

    def test_embed_count_mismatch_raises(self):
        from src.ollama.client import OllamaError, embed

        with mock.patch("src.ollama.client._request_json", return_value={"data": [{"embedding": [0.1]}]}):
            with self.assertRaises(OllamaError):
                embed("http://localhost:11434/v1", "nomic-embed-text", ["a", "b"])


class TestModelsCommand(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.config = Config(
            {
                "ollama": {"host": "http://localhost:11434"},
                "llm": {"provider": "ollama", "model": "gemma3:12b-it-qat"},
                "embedding": {"provider": "ollama", "model": "nomic-embed-text:latest"},
                "index": {"path": str(self.root / "index")},
            },
            path=self.root / "config.yaml",
        )
        self.app = App(self.config)

    def tearDown(self):
        self._tmp.cleanup()

    def test_help_lists_models_command(self):
        out = build_help()
        self.assertIn("/models", out)
        self.assertIn("set embedding", build_help("/models"))

    def test_models_list_marks_current(self):
        with mock.patch.object(main, "ollama_list_models", return_value=_fake_models()):
            _, out = handle_command("/models", self.app)
        self.assertIn("gemma3:12b-it-qat", out)
        self.assertIn("nomic-embed-text:latest", out)
        self.assertIn("LLM", out)
        self.assertIn("EMB", out)

    def test_models_set_llm_persists_override(self):
        with mock.patch.object(main, "ollama_list_models", return_value=_fake_models()):
            _, out = handle_command("/models set llm gemma3:12b-it-qat", self.app)
        self.assertIn("переключена", out)
        overrides = self.root / "config.local.json"
        self.assertTrue(overrides.exists())
        self.assertEqual(self.app.config.llm_model, "gemma3:12b-it-qat")

    def test_models_set_embedding_accepts_embedding_model(self):
        with mock.patch.object(main, "ollama_list_models", return_value=_fake_models()):
            _, out = handle_command("/models set embedding nomic-embed-text:latest", self.app)
        self.assertIn("переиндексируйте", out)
        self.assertEqual(self.app.config.embedding_model, "nomic-embed-text:latest")

    def test_models_set_embedding_rejects_non_embedding_model(self):
        with mock.patch.object(main, "ollama_list_models", return_value=_fake_models()):
            with self.assertRaises(ValueError):
                handle_command("/models set embedding gemma3:12b-it-qat", self.app)

    def test_models_set_unknown_model_raises(self):
        with mock.patch.object(main, "ollama_list_models", return_value=_fake_models()):
            with self.assertRaises(ValueError):
                handle_command("/models set llm does-not-exist", self.app)


if __name__ == "__main__":
    unittest.main()
