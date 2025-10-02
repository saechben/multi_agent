"""Standalone A2A server exposing the division MCP agent."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from agent_system.arithmetic_a2a import DivisionAgentExecutor


def build_application() -> A2AStarletteApplication:
    skill = AgentSkill(
        id="division",
        name="Division",
        description="Divides numbers via the MCP division tool",
        tags=["math", "division", "quotient"],
        examples=["Divide 20 by 4", "Compute 9 / 3"],
    )

    agent_card = AgentCard(
        name="MCP Division Agent",
        description="Provides division by delegating to the MCP division tool",
        url=os.getenv("A2A_PUBLIC_URL", "http://localhost:9997/"),
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[skill],
        version="1.0.0",
        capabilities=AgentCapabilities(),
    )

    request_handler = DefaultRequestHandler(
        agent_executor=DivisionAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    return A2AStarletteApplication(
        http_handler=request_handler,
        agent_card=agent_card,
    )


def main() -> None:
    host = os.getenv("A2A_HOST", "0.0.0.0")
    port = int(os.getenv("A2A_PORT", "9997"))
    app = build_application().build()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
