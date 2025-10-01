"""A2A server exposing the arithmetic host orchestrator."""

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
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from agent_system.arithmetic_a2a import ArithmeticHostAgentExecutor
from agent_system.logging_task_store import FileTaskStore


def build_application() -> A2AStarletteApplication:
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
        url=os.getenv("A2A_HOST_PUBLIC_URL", "http://localhost:10010/"),
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[skill],
        version="1.0.0",
        capabilities=AgentCapabilities(),
    )

    request_handler = DefaultRequestHandler(
        agent_executor=ArithmeticHostAgentExecutor(),
        task_store=FileTaskStore(),
    )

    return A2AStarletteApplication(http_handler=request_handler, agent_card=agent_card)


def main() -> None:
    host = os.getenv("A2A_HOST_HOST", "0.0.0.0")
    port = int(os.getenv("A2A_HOST_PORT", "10010"))
    app = build_application().build()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
