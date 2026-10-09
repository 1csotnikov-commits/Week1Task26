"""Загрузка и работа с конфигурацией (config/config.yaml)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
# Файл локальных переопределений (создаётся командой /models, не коммитится в git).
LOCAL_OVERRIDES_NAME = "config.local.json"


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Рекурсивно слить patch в base (значения patch имеют приоритет)."""
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


class Config:
    """Обёртка над YAML-конфигом с доступом по точечному пути (dotted path)."""

    def __init__(self, data: Dict[str, Any], path: Path = DEFAULT_CONFIG_PATH):
        self.data = data or {}
        self.path = path

    @classmethod
    def load(cls, path: Optional[str | Path] = None) -> "Config":
        config_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not config_path.exists():
            raise FileNotFoundError(
                f"Файл конфигурации не найден: {config_path}. "
                "Создайте config/config.yaml или укажите путь через --config."
            )
        with open(config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        # Накладываем локальные переопределения (выбор моделей через /models и т.п.).
        overrides_path = config_path.with_name(LOCAL_OVERRIDES_NAME)
        if overrides_path.exists():
            overrides = cls._read_json(overrides_path)
            if isinstance(overrides, dict):
                _deep_merge(data, overrides)

        return cls(data, config_path)

    @staticmethod
    def _read_json(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    @property
    def local_overrides_path(self) -> Path:
        """Путь к файлу локальных переопределений (config.local.json)."""
        base = self.path if self.path else DEFAULT_CONFIG_PATH
        return Path(base).with_name(LOCAL_OVERRIDES_NAME)

    def save_local_overrides(self, patch: Dict[str, Any]) -> Path:
        """Сохранить переопределения в config.local.json и применить их в памяти."""
        path = self.local_overrides_path
        existing = self._read_json(path)
        if not isinstance(existing, dict):
            existing = {}
        _deep_merge(existing, patch)
        _deep_merge(self.data, patch)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def get(self, dotted: str, default: Any = None) -> Any:
        """Получить значение по точечному пути, например 'chunking.fixed_size.chunk_size'."""
        current: Any = self.data
        for part in dotted.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def resolve_path(self, value: str | Path) -> Path:
        """Преобразовать путь из конфига в абсолютный (относительно корня проекта)."""
        p = Path(value)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def documents_path(self) -> Path:
        return self.resolve_path(self.get("documents.path", "data/documents"))

    @property
    def index_path(self) -> Path:
        return self.resolve_path(self.get("index.path", "data/index"))

    @property
    def embedding_model(self) -> str:
        return self.get("embedding.model", "nomic-ai/nomic-embed-text-v2-moe")

    @property
    def embedding_provider(self) -> str:
        return self.get("embedding.provider", "nomic")

    @property
    def embedding_batch_size(self) -> int:
        return int(self.get("embedding.batch_size", 16))

    @property
    def default_strategy(self) -> str:
        return self.get("chunking.default_strategy", "structural")

    @property
    def document_extensions(self) -> list:
        return list(self.get("documents.extensions", [".md", ".txt", ".py", ".js", ".json"]))

    @property
    def ollama_host(self) -> str:
        return self.get("ollama.host", "http://localhost:11434")

    @property
    def llm_provider(self) -> str:
        return self.get("llm.provider", "ollama")

    @property
    def llm_model(self) -> str:
        return self.get("llm.model", "")

    @property
    def embedding_base_url(self) -> str:
        explicit = self.get("embedding.base_url", "")
        if explicit:
            return explicit
        return f"{str(self.ollama_host).rstrip('/')}/v1"
