"""Integration test that spins up MCP + A2A arithmetic agents and evaluates an expression."""

from __future__ import annotations

import asyncio
import os
from typing import Iterable

import httpx
import uvicorn
from dotenv import load_dotenv

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    JSONRPCErrorResponse,
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
    SendMessageSuccessResponse,
    Task,
    TextPart,
)

from agent_system.MCP import mcp
from agent_system.a2a_server_addition import build_application as build_addition_app
from agent_system.a2a_server_host import build_application as build_host_app
from agent_system.a2a_server_subtraction import build_application as build_subtraction_app


async def start_uvicorn(app, host: str, port: int) -> tuple[uvicorn.Server, asyncio.Task[None]]:
    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    # Wait until the server reports it has started
    while not server.started:  # type: ignore[attr-defined]
        await asyncio.sleep(0.05)
    return server, task


async def stop_uvicorn(server: uvicorn.Server, task: asyncio.Task[None]) -> None:
    server.should_exit = True
    await task


def ensure_env(var: str) -> str:
    value = os.getenv(var)
    if not value:
        raise RuntimeError(f"Environment variable {var} must be set for this integration test")
    return value


def _message_to_text(message: Message) -> str:
    texts: list[str] = []
    for part in message.parts:
        root = part.root
        if isinstance(root, TextPart):
            texts.append(root.text)
    return "\n".join(texts)


async def run_integration() -> None:
    # Make environment variables from .env available
    load_dotenv()
    ensure_env("LLM_API_KEY")

    host = "127.0.0.1"
    mcp_port = 18200
    addition_port = 18201
    subtraction_port = 18202
    host_port = 18203

    os.environ["MCP_SERVER_URL"] = f"http://{host}:{mcp_port}/mcp"
    os.environ["A2A_PUBLIC_URL"] = f"http://{host}:{addition_port}"
    os.environ["ADDITION_AGENT_URL"] = f"http://{host}:{addition_port}"
    os.environ["SUBTRACTION_AGENT_URL"] = f"http://{host}:{subtraction_port}"
    os.environ["A2A_HOST_PUBLIC_URL"] = f"http://{host}:{host_port}"

    mcp_app = mcp.streamable_http_app()
    addition_app = build_addition_app().build()
    subtraction_app = build_subtraction_app().build()
    host_app = build_host_app().build()

    servers: list[tuple[uvicorn.Server, asyncio.Task[None]]] = []
    try:
        servers.append(await start_uvicorn(mcp_app, host, mcp_port))
        servers.append(await start_uvicorn(addition_app, host, addition_port))
        servers.append(await start_uvicorn(subtraction_app, host, subtraction_port))
        servers.append(await start_uvicorn(host_app, host, host_port))

        async with httpx.AsyncClient() as http_client:
            resolver = A2ACardResolver(httpx_client=http_client, base_url=f"http://{host}:{host_port}")
            card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=http_client, agent_card=card)

            message = Message(
                role=Role.user,
                messageId="test",
                parts=[Part(root=TextPart(text="Compute 10 - 3 + 2"))],
            )
            request = SendMessageRequest(
                id="request-1",
                params=MessageSendParams(message=message),
            )
            try:
                response = await client.send_message(request, http_kwargs={"timeout": 300})
            except Exception as exc:
                raise RuntimeError(f"A2A host request failed: {exc}") from exc

            payload = response.root
            if isinstance(payload, JSONRPCErrorResponse):
                raise RuntimeError(f"Host agent returned error: {payload.error.message}")

            assert isinstance(payload, SendMessageSuccessResponse)
            result = payload.result
            final_text: str
            if isinstance(result, Task):
                history: Iterable[Message] = result.history or []
                agent_messages = [msg for msg in history if msg.role == Role.agent]
                if not agent_messages:
                    raise RuntimeError("Host agent did not emit any messages")
                final_text = _message_to_text(agent_messages[-1])
            else:
                final_text = _message_to_text(result)

            print("Host agent final reply:\n", final_text)
            assert "9" in final_text

    finally:
        for server, task in reversed(servers):
            await stop_uvicorn(server, task)


if __name__ == "__main__":
    asyncio.run(run_integration())
