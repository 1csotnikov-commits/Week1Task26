"""Клиент локального сервера Ollama (нативный и OpenAI-совместимый API)."""

from __future__ import annotations

from src.ollama.client import (
    DEFAULT_HOST,
    OllamaError,
    chat,
    embed,
    list_models,
    pull_model,
)

__all__ = [
    "DEFAULT_HOST",
    "OllamaError",
    "chat",
    "embed",
    "list_models",
    "pull_model",
]
