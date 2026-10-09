"""LLM-провайдер для локальных моделей Ollama (нативный ``/api/chat``).

Модель выбирается из установленных в Ollama (см. команду ``/models``).
Параметры генерации (``temperature``, ``num_ctx``, ``num_predict`` и т.д.)
передаются в ``options`` нативного API, поэтому доступны все настройки Ollama.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from src.llm.base import LLMProvider, LLMResponse
from src.ollama.client import DEFAULT_HOST, OllamaError, chat

DEFAULT_MODEL = "gemma3:12b-it-qat"


class OllamaProvider(LLMProvider):
    """LLM-провайдер поверх локального сервера Ollama."""

    name = "ollama"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        timeout: float = 300.0,
        num_ctx: Optional[int] = 8192,
        keep_alive: Optional[str] = "5m",
        top_p: Optional[float] = None,
    ):
        self.model = model
        self.host = host
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.top_p = top_p

    def _options(self, temperature: Optional[float], max_tokens: Optional[int]) -> Dict[str, Any]:
        options: Dict[str, Any] = {
            "temperature": self.temperature if temperature is None else temperature,
            "num_predict": self.max_tokens if max_tokens is None else max_tokens,
        }
        if self.num_ctx:
            options["num_ctx"] = int(self.num_ctx)
        if self.top_p is not None:
            options["top_p"] = self.top_p
        return options

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        started = time.time()
        try:
            data = chat(
                host=self.host,
                model=self.model,
                messages=messages,
                options=self._options(temperature, max_tokens),
                keep_alive=self.keep_alive,
                timeout=self.timeout if timeout is None else timeout,
            )
        except OllamaError as exc:
            raise RuntimeError(f"Ошибка вызова Ollama: {exc}") from exc

        elapsed_ms = int((time.time() - started) * 1000)
        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        completion_tokens = int(data.get("eval_count") or 0)
        message = data.get("message") or {}
        return LLMResponse(
            text=(message.get("content") or "").strip(),
            model=data.get("model") or self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            elapsed_ms=elapsed_ms,
        )
