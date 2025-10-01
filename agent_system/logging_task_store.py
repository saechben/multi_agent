"""TaskStore implementation that persists tasks as JSON files under a logs directory."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from a2a.server.context import ServerCallContext
from a2a.server.tasks.task_store import TaskStore
from a2a.types import Task


class FileTaskStore(TaskStore):
    """Persists each task as ``logs/<task_id>.json`` for offline inspection."""

    def __init__(self, root: str | Path = "logs") -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _path_for(self, task_id: str) -> Path:
        return self._root / f"{task_id}.json"

    async def save(self, task: Task, context: ServerCallContext | None = None) -> None:
        data = task.model_dump(mode="json")
        payload = json.dumps(data, separators=(",", ":"))
        async with self._lock:
            self._path_for(task.id).write_text(payload, encoding="utf-8")

    async def get(self, task_id: str, context: ServerCallContext | None = None) -> Task | None:
        path = self._path_for(task_id)
        async with self._lock:
            if not path.exists():
                return None
            raw = path.read_text(encoding="utf-8")
        data: Any = json.loads(raw)
        return Task.model_validate(data)

    async def delete(self, task_id: str, context: ServerCallContext | None = None) -> None:
        path = self._path_for(task_id)
        async with self._lock:
            if path.exists():
                path.unlink()
