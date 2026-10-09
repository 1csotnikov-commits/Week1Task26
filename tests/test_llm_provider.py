"""Тесты LLM-провайдеров: Ollama (основной) и DeepSeek (legacy)."""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm import make_llm_provider  # noqa: E402
from src.llm.deepseek_provider import DeepSeekProvider  # noqa: E402
from src.llm.ollama_provider import OllamaProvider  # noqa: E402
from src.ollama.client import OllamaError  # noqa: E402


class TestDeepSeekProvider(unittest.TestCase):
    def test_missing_api_key_raises_clear_error(self):
        provider = DeepSeekProvider(api_key="")
        with self.assertRaises(RuntimeError) as ctx:
            provider.complete([{"role": "user", "content": "привет"}])
        self.assertIn("DEEPSEEK_API_KEY", str(ctx.exception))


class TestOllamaProvider(unittest.TestCase):
    def test_complete_maps_response_and_passes_options(self):
        fake = {
            "model": "gemma3:12b-it-qat",
            "message": {"role": "assistant", "content": "  привет  "},
            "prompt_eval_count": 12,
            "eval_count": 3,
        }
        provider = OllamaProvider(model="gemma3:12b-it-qat", num_ctx=4096, max_tokens=1024)
        with mock.patch("src.llm.ollama_provider.chat", return_value=fake) as patched:
            resp = provider.complete([{"role": "user", "content": "привет"}])

        self.assertEqual(resp.text, "привет")
        self.assertEqual(resp.model, "gemma3:12b-it-qat")
        self.assertEqual(resp.prompt_tokens, 12)
        self.assertEqual(resp.completion_tokens, 3)
        self.assertEqual(resp.total_tokens, 15)

        _, kwargs = patched.call_args
        self.assertEqual(kwargs["model"], "gemma3:12b-it-qat")
        self.assertEqual(kwargs["options"]["num_ctx"], 4096)
        self.assertEqual(kwargs["options"]["num_predict"], 1024)

    def test_connection_error_is_wrapped(self):
        provider = OllamaProvider()
        with mock.patch("src.llm.ollama_provider.chat", side_effect=OllamaError("нет соединения")):
            with self.assertRaises(RuntimeError) as ctx:
                provider.complete([{"role": "user", "content": "привет"}])
        self.assertIn("Ollama", str(ctx.exception))


class _FakeConfig:
    """Мини-конфиг для проверки фабрики (duck-typing: нужен только get)."""

    def __init__(self, values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


class TestMakeLLMProvider(unittest.TestCase):
    def test_default_is_ollama(self):
        config = _FakeConfig({
            "llm.model": "gpt-oss:20b",
            "llm.host": "",
            "ollama.host": "http://localhost:11434",
        })
        provider = make_llm_provider(config)
        self.assertIsInstance(provider, OllamaProvider)
        self.assertEqual(provider.model, "gpt-oss:20b")
        self.assertEqual(provider.host, "http://localhost:11434")

    def test_unknown_provider_raises(self):
        config = _FakeConfig({"llm.provider": "unknown"})
        with self.assertRaises(ValueError):
            make_llm_provider(config)


if __name__ == "__main__":
    unittest.main()

