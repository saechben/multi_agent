"""Standalone A2A server exposing the subtraction MCP agent."""

from __future__ import annotations

import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from agent_system.arithmetic_a2a import SubtractionAgentExecutor


def build_application() -> A2AStarletteApplication:
    skill = AgentSkill(
        id="subtraction",
        name="Subtraction",
        description="Subtracts numbers via the MCP subtraction tool",
        tags=["math", "subtraction", "difference"],
        examples=["Subtract 7 from 20", "What is 42 minus 19?"],
    )

    agent_card = AgentCard(
        name="MCP Subtraction Agent",
        description="Provides subtraction by delegating to the MCP subtraction tool",
        url=os.getenv("A2A_PUBLIC_URL", "http://localhost:10000/"),
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[skill],
        version="1.0.0",
        capabilities=AgentCapabilities(),
    )

    request_handler = DefaultRequestHandler(
        agent_executor=SubtractionAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    return A2AStarletteApplication(
        http_handler=request_handler,
        agent_card=agent_card,
    )


def main() -> None:
    host = os.getenv("A2A_HOST", "0.0.0.0")
    port = int(os.getenv("A2A_PORT", "10000"))
    app = build_application().build()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
