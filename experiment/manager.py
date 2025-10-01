"""Experiment lifecycle management for the arithmetic agent stack."""

from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path

from .config import ExperimentConfig
from .host_client import HostClient, HostExecution
from .servers import (
    ServerHandle,
    build_addition_app,
    build_host_app,
    build_mcp_app,
    build_subtraction_app,
    start_uvicorn,
    stop_uvicorn,
)


class Experiment:
    """Manages the lifecycle of the arithmetic agent cluster for experiments."""

    def __init__(self, config: ExperimentConfig | None = None) -> None:
        self.config = config or ExperimentConfig()
        self.config.log_root = Path(self.config.log_root)
        self._handles: list[ServerHandle] = []
        self._env_overrides: dict[str, str | None] = {}
        self._host_client: HostClient | None = None
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        cfg = self.config
        cfg.log_root.mkdir(parents=True, exist_ok=True)

        if cfg.llm_api_key:
            self._set_env("LLM_API_KEY", cfg.llm_api_key)
        elif not os.getenv("LLM_API_KEY"):
            raise RuntimeError("LLM_API_KEY must be set to launch the experiment")
        self._set_env("LLM_MODEL", cfg.llm_model)

        base = f"http://{cfg.host}"
        mcp_url = f"{base}:{cfg.mcp_port}/mcp"
        addition_url = f"{base}:{cfg.addition_port}"
        subtraction_url = f"{base}:{cfg.subtraction_port}"
        host_url = f"{base}:{cfg.host_port}"

        self._set_env("MCP_SERVER_URL", mcp_url)
        self._set_env("ADDITION_AGENT_URL", addition_url)
        self._set_env("SUBTRACTION_AGENT_URL", subtraction_url)
        self._set_env("A2A_HOST_PUBLIC_URL", host_url)

        mcp_app = build_mcp_app()
        addition_app = build_addition_app(addition_url, cfg.log_root / "addition").build()
        subtraction_app = build_subtraction_app(subtraction_url, cfg.log_root / "subtraction").build()
        host_app = build_host_app(host_url, cfg.log_root / "host").build()

        started: list[ServerHandle] = []
        try:
            started.append(await start_uvicorn(mcp_app, cfg.host, cfg.mcp_port, "mcp"))
            started.append(await start_uvicorn(addition_app, cfg.host, cfg.addition_port, "addition"))
            started.append(await start_uvicorn(subtraction_app, cfg.host, cfg.subtraction_port, "subtraction"))
            started.append(await start_uvicorn(host_app, cfg.host, cfg.host_port, "host"))
        except Exception:
            for handle in reversed(started):
                with contextlib.suppress(Exception):
                    await stop_uvicorn(handle)
            self._restore_env()
            raise

        self._handles.extend(started)
        self._host_client = HostClient(host_url)
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        if self._host_client is not None:
            await self._host_client.aclose()
            self._host_client = None
        while self._handles:
            handle = self._handles.pop()
            try:
                await stop_uvicorn(handle)
            except Exception:
                continue
        self._restore_env()
        self._started = False

    async def __aenter__(self) -> "Experiment":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    async def evaluate(self, expression: str) -> HostExecution:
        if not self._started or self._host_client is None:
            raise RuntimeError("Experiment has not been started")
        return await self._host_client.evaluate(expression)

    def _set_env(self, key: str, value: str) -> None:
        if key not in self._env_overrides:
            self._env_overrides[key] = os.environ.get(key)
        os.environ[key] = value

    def _restore_env(self) -> None:
        for key, previous in reversed(list(self._env_overrides.items())):
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        self._env_overrides.clear()


async def run_single_expression(expression: str, config: ExperimentConfig | None = None) -> HostExecution:
    """Helper to run one-off expressions without managing the experiment manually."""

    async with Experiment(config) as experiment:
        return await experiment.evaluate(expression)


def run(expression: str, config: ExperimentConfig | None = None) -> HostExecution:
    """Synchronous wrapper for quick experiments."""

    return asyncio.run(run_single_expression(expression, config))


__all__ = ["Experiment", "run", "run_single_expression"]
