"""Integration test that spins up arithmetic agents and measures their latency directly."""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from decimal import Decimal
from typing import Iterable

import uvicorn
from dotenv import load_dotenv

from agent_system.MCP import mcp
from agent_system.arithmetic_a2a.clients import AdditionClient, SubtractionClient
from agent_system.arithmetic_a2a.remote_executors import format_decimal
from agent_system.a2a_server_addition import build_application as build_addition_app
from agent_system.a2a_server_subtraction import build_application as build_subtraction_app


async def start_uvicorn(app, host: str, port: int) -> tuple[uvicorn.Server, asyncio.Task[None]]:
    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:  # type: ignore[attr-defined]
        await asyncio.sleep(0.05)
    return server, task


async def stop_uvicorn(server: uvicorn.Server, task: asyncio.Task[None], *, timeout: float = 5.0) -> None:
    """Request shutdown and ensure the server task terminates promptly."""

    server.should_exit = True
    pending_exc: Exception | None = None
    try:
        await asyncio.wait_for(task, timeout=timeout)
        return
    except asyncio.TimeoutError:
        server.force_exit = True
    except Exception as exc:  # pragma: no cover - defensive path
        server.force_exit = True
        pending_exc = exc

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    if pending_exc is not None:
        raise pending_exc


def ensure_env(var: str) -> str:
    value = os.getenv(var)
    if not value:
        raise RuntimeError(f"Environment variable {var} must be set for this integration test")
    return value


async def call_arithmetic_agents(initial_value: Decimal, steps: Iterable[tuple[str, Decimal]]) -> Decimal:
    """Mimic the host agent by delegating to the remote addition/subtraction agents only."""

    addition_client = AdditionClient(os.environ["ADDITION_AGENT_URL"])
    subtraction_client = SubtractionClient(os.environ["SUBTRACTION_AGENT_URL"])
    try:
        current = initial_value
        for idx, (operation, operand) in enumerate(steps, start=1):
            operator_symbol = "+" if operation == "add" else "-"
            start = time.perf_counter()
            if operation == "add":
                next_value = await addition_client.add(current, operand)
            elif operation == "sub":
                next_value = await subtraction_client.subtract(current, operand)
            else:
                raise ValueError(f"Unsupported operation '{operation}'")
            elapsed_ms = (time.perf_counter() - start) * 1_000
            print(
                f"Direct step {idx}: {format_decimal(current)} {operator_symbol} "
                f"{format_decimal(operand)} = {format_decimal(next_value)} ({elapsed_ms:.1f} ms)"
            )
            current = next_value
        return current
    finally:
        await addition_client.shutdown()
        await subtraction_client.shutdown()


async def run_integration() -> None:
    load_dotenv()
    ensure_env("LLM_API_KEY")

    host = "127.0.0.1"
    mcp_port = 18200
    addition_port = 18201
    subtraction_port = 18202

    os.environ["MCP_SERVER_URL"] = f"http://{host}:{mcp_port}/mcp"
    os.environ["ADDITION_AGENT_URL"] = f"http://{host}:{addition_port}"
    os.environ["SUBTRACTION_AGENT_URL"] = f"http://{host}:{subtraction_port}"

    mcp_app = mcp.streamable_http_app()
    os.environ["A2A_PUBLIC_URL"] = f"http://{host}:{addition_port}"
    addition_app = build_addition_app().build()
    os.environ["A2A_PUBLIC_URL"] = f"http://{host}:{subtraction_port}"
    subtraction_app = build_subtraction_app().build()

    servers: list[tuple[uvicorn.Server, asyncio.Task[None]]] = []
    try:
        servers.append(await start_uvicorn(mcp_app, host, mcp_port))
        servers.append(await start_uvicorn(addition_app, host, addition_port))
        servers.append(await start_uvicorn(subtraction_app, host, subtraction_port))

        steps: list[tuple[str, Decimal]] = [
            ("sub", Decimal("3")),
            ("add", Decimal("2")),
        ]
        total_start = time.perf_counter()
        direct_result = await call_arithmetic_agents(Decimal("10"), steps)
        total_elapsed_ms = (time.perf_counter() - total_start) * 1_000
        print("Direct arithmetic agents final result:\n", format_decimal(direct_result))
        print(f"Total arithmetic latency: {total_elapsed_ms:.1f} ms")
        assert direct_result == Decimal("9")
    finally:
        for server, task in reversed(servers):
            await stop_uvicorn(server, task)


def debug_experiment(expression: str) -> None:
    """Helper for VS Code debugger to execute an experiment run."""

    from experiment import run

    outcome = run(expression)
    print("Expression:", outcome.expression)
    print("Final value:", outcome.final_value)
    print("Agent messages:")
    for line in outcome.messages:
        print("-", line)
    if outcome.task_id:
        print("Task ID:", outcome.task_id)

if __name__ == "__main__":
    debug_experiment("10+5-3+6")


