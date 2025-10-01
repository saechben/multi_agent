"""Helpers for standing up the MCP and arithmetic A2A servers."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from agent_system.MCP import mcp
from agent_system.arithmetic_a2a import (
    AdditionAgentExecutor,
    ArithmeticHostAgentExecutor,
    SubtractionAgentExecutor,
)
from agent_system.logging_task_store import FileTaskStore


@dataclass
class ServerHandle:
    """Tracks a running uvicorn server."""

    name: str
    server: uvicorn.Server
    task: asyncio.Task[None]


async def start_uvicorn(app: Any, host: str, port: int, name: str) -> ServerHandle:
    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not getattr(server, "started", False):
        await asyncio.sleep(0.05)
    return ServerHandle(name=name, server=server, task=task)


async def stop_uvicorn(handle: ServerHandle, timeout: float = 5.0) -> None:
    handle.server.should_exit = True
    exc: Exception | None = None
    try:
        await asyncio.wait_for(handle.task, timeout=timeout)
    except asyncio.TimeoutError:
        handle.server.force_exit = True
        exc = asyncio.TimeoutError()
    except Exception as err:
        handle.server.force_exit = True
        exc = err
    finally:
        if not handle.task.done():
            handle.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await handle.task
    if exc is not None:
        raise exc


def build_mcp_app() -> Any:
    return mcp.streamable_http_app()


def build_addition_app(public_url: str, log_dir: Path) -> A2AStarletteApplication:
    skill = AgentSkill(
        id="addition",
        name="Addition",
        description="Adds numbers via the MCP addition tool",
        tags=["math", "addition", "sum"],
        examples=["Add 12 and 30", "What is the sum of 5 and 9?"],
    )
    agent_card = AgentCard(
        name="MCP Addition Agent",
        description="Provides addition by delegating to the MCP addition tool",
        url=public_url,
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[skill],
        version="1.0.0",
        capabilities=AgentCapabilities(),
    )
    handler = DefaultRequestHandler(
        agent_executor=AdditionAgentExecutor(),
        task_store=FileTaskStore(log_dir),
    )
    return A2AStarletteApplication(http_handler=handler, agent_card=agent_card)


def build_subtraction_app(public_url: str, log_dir: Path) -> A2AStarletteApplication:
    skill = AgentSkill(
        id="subtraction",
        name="Subtraction",
        description="Subtracts numbers via the MCP subtraction tool",
        tags=["math", "subtraction", "difference"],
        examples=["Subtract 7 from 20", "Compute 50 - 13"],
    )
    agent_card = AgentCard(
        name="MCP Subtraction Agent",
        description="Provides subtraction by delegating to the MCP subtraction tool",
        url=public_url,
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[skill],
        version="1.0.0",
        capabilities=AgentCapabilities(),
    )
    handler = DefaultRequestHandler(
        agent_executor=SubtractionAgentExecutor(),
        task_store=FileTaskStore(log_dir),
    )
    return A2AStarletteApplication(http_handler=handler, agent_card=agent_card)


def build_host_app(public_url: str, log_dir: Path) -> A2AStarletteApplication:
    skill = AgentSkill(
        id="arithmetic_planning",
        name="Arithmetic Planner",
        description="Breaks expressions into addition/subtraction subtasks",
        tags=["math", "planner", "arithmetic"],
        examples=["Evaluate 10 - 3 + 2", "Compute 50 + 25 - 5"],
    )
    agent_card = AgentCard(
        name="Arithmetic Host Agent",
        description="Routes expressions to MCP addition and subtraction agents",
        url=public_url,
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[skill],
        version="1.0.0",
        capabilities=AgentCapabilities(),
    )
    handler = DefaultRequestHandler(
        agent_executor=ArithmeticHostAgentExecutor(),
        task_store=FileTaskStore(log_dir),
    )
    return A2AStarletteApplication(http_handler=handler, agent_card=agent_card)


__all__ = [
    "ServerHandle",
    "start_uvicorn",
    "stop_uvicorn",
    "build_mcp_app",
    "build_addition_app",
    "build_subtraction_app",
    "build_host_app",
]
