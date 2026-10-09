"""Низкоуровневый клиент локального сервера Ollama.

Использует только стандартную библиотеку (urllib), поэтому не зависит от версий
сторонних пакетов (Python 3.14 совместим). Поддерживает:

- ``chat``   — нативный ``POST /api/chat`` (полный контроль над ``options`` и ``keep_alive``);
- ``embed``  — OpenAI-совместимый ``POST /v1/embeddings`` (работает, даже если нативный
  ``/api/embed`` отключён в сборке сервера);
- ``list_models`` / ``pull_model`` — просмотр и загрузка установленных моделей.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_HOST = "http://localhost:11434"


class OllamaError(RuntimeError):
    """Ошибка обращения к локальному серверу Ollama."""


def _base(host: str) -> str:
    return (host or DEFAULT_HOST).rstrip("/")


def _request_json(url: str, payload: Optional[Dict[str, Any]], timeout: float) -> Dict[str, Any]:
    """Выполнить GET (payload=None) или POST и вернуть разобранный JSON."""
    data = None
    method = "GET"
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        method = "POST"
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:  # сервер ответил кодом ошибки
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:  # noqa: BLE001 - тело ошибки недоступно
            detail = ""
        if exc.code == 501 or "does not support embeddings" in detail:
            detail += (
                "\nПодсказка: используйте эндпоинт /v1/embeddings (OpenAI-совместимый) "
                "с embedding-моделью, например nomic-embed-text."
            )
        raise OllamaError(f"Ollama вернула HTTP {exc.code}: {detail or 'без тела'}") from exc
    except urllib.error.URLError as exc:
        raise OllamaError(
            f"Не удалось подключиться к Ollama по адресу {url}. "
            "Убедитесь, что сервер запущен (ollama serve). "
            f"Техническая ошибка: {exc.reason}"
        ) from exc
    except TimeoutError as exc:  # socket.timeout в части версий
        raise OllamaError(f"Истёк таймаут ожидания ответа Ollama ({url}).") from exc

    if not body.strip():
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise OllamaError(f"Некорректный JSON от Ollama: {body[:200]!r}") from exc


def chat(
    host: str,
    model: str,
    messages: Sequence[Dict[str, str]],
    options: Optional[Dict[str, Any]] = None,
    keep_alive: Optional[str] = None,
    timeout: float = 300.0,
) -> Dict[str, Any]:
    """Отправить сообщения в ``/api/chat`` (без стриминга) и вернуть сырой ответ."""
    payload: Dict[str, Any] = {"model": model, "messages": list(messages), "stream": False}
    if options:
        payload["options"] = {k: v for k, v in options.items() if v is not None}
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    return _request_json(f"{_base(host)}/api/chat", payload, timeout)


def embed(
    base_url: str,
    model: str,
    inputs: Sequence[str],
    timeout: float = 300.0,
) -> List[List[float]]:
    """Получить эмбеддинги через OpenAI-совместимый ``/v1/embeddings``."""
    if not inputs:
        return []
    payload = {"model": model, "input": list(inputs)}
    data = _request_json(f"{_base(base_url)}/embeddings", payload, timeout)
    items = data.get("data") or []
    if len(items) != len(inputs):
        raise OllamaError(
            f"Ollama вернула {len(items)} эмбеддингов вместо {len(inputs)}."
        )
    try:
        return [list(item["embedding"]) for item in items]
    except (KeyError, TypeError) as exc:
        raise OllamaError("Ответ /v1/embeddings не содержит поле embedding.") from exc


def list_models(host: str, timeout: float = 15.0) -> List[Dict[str, Any]]:
    """Список установленных моделей (``GET /api/tags``) с возможностями."""
    data = _request_json(f"{_base(host)}/api/tags", None, timeout)
    result: List[Dict[str, Any]] = []
    for item in data.get("models") or []:
        details = item.get("details") or {}
        result.append(
            {
                "name": item.get("name") or item.get("model") or "",
                "size": int(item.get("size") or 0),
                "capabilities": list(item.get("capabilities") or []),
                "parameter_size": details.get("parameter_size") or "",
                "quantization": details.get("quantization_level") or "",
                "family": details.get("family") or "",
            }
        )
    return result


def pull_model(host: str, name: str, timeout: float = 3600.0) -> Dict[str, Any]:
    """Загрузить модель (``POST /api/pull``), дождавшись завершения."""
    return _request_json(f"{_base(host)}/api/pull", {"model": name, "stream": False}, timeout)
