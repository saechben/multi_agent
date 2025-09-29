"""Executors that expose MCP arithmetic tools over the A2A AgentExecutor interface."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from decimal import Decimal

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.utils import new_agent_text_message
from langchain_openai import ChatOpenAI

from ..Agent import LangGraphReactAgent, ReactionStep


@dataclass
class _MCPAgentRunner:
    """Serialises access to a LangGraph MCP agent so calls do not overlap."""

    name: str
    tool_name: str
    prompt: str
    server_url: str
    llm: ChatOpenAI

    def __post_init__(self) -> None:
        self._agent = LangGraphReactAgent(
            name=self.name,
            server_url=self.server_url,
            llm=self.llm,
            prompt=self.prompt,
            tool_names=[self.tool_name],
        )
        self._lock = asyncio.Lock()

    async def run(self, expression: str) -> tuple[str, tuple[ReactionStep, ...]]:
        async with self._lock:
            await self._agent.initialize()
            result = await self._agent.handle_task({"description": expression})
            trace = tuple(self._agent.trace)
        return result, trace

    async def shutdown(self) -> None:
        async with self._lock:
            await self._agent.shutdown()


class MCPToolAgentExecutor(AgentExecutor):
    """Base executor that delegates arithmetic to a single MCP tool."""

    def __init__(
        self,
        *,
        tool_name: str,
        prompt: str,
        llm: ChatOpenAI | None = None,
        server_url: str | None = None,
    ) -> None:
        if llm is None:
            api_key = os.getenv("LLM_API_KEY")
            if not api_key:
                raise RuntimeError("LLM_API_KEY must be set for MCP Tool executors")
            model_id = os.getenv("LLM_MODEL", "gpt-4o-mini")
            llm = ChatOpenAI(model=model_id, api_key=api_key, temperature=0.0)

        resolved_url = server_url or os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")
        self._runner = _MCPAgentRunner(
            name=f"{tool_name}-agent",
            tool_name=tool_name,
            prompt=prompt,
            server_url=resolved_url,
            llm=llm,
        )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        expression = (context.get_user_input() or "").strip()
        if not expression:
            await event_queue.enqueue_event(
                new_agent_text_message("Please provide an expression to evaluate."),
            )
            return

        try:
            result, trace = await self._runner.run(expression)
        except Exception as exc:  # pragma: no cover - defensive guard
            await event_queue.enqueue_event(
                new_agent_text_message(f"Failed to process request: {exc}"),
            )
            return

        await event_queue.enqueue_event(
            new_agent_text_message(_format_trace(result, trace)),
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(new_agent_text_message("Cancellation not supported."))

    async def shutdown(self) -> None:
        await self._runner.shutdown()


class AdditionAgentExecutor(MCPToolAgentExecutor):
    """Executor that forces the agent to call the MCP addition tool."""

    PROMPT = (
        "You specialise in addition. Always call the 'add' MCP tool to compute"
        " sums and never perform arithmetic yourself."
    )

    def __init__(self, *, llm: ChatOpenAI | None = None, server_url: str | None = None) -> None:
        super().__init__(tool_name="add", prompt=self.PROMPT, llm=llm, server_url=server_url)


class SubtractionAgentExecutor(MCPToolAgentExecutor):
    """Executor that forces the agent to call the MCP subtraction tool."""

    PROMPT = (
        "You specialise in subtraction. Always call the 'sub' MCP tool to"
        " compute differences and never perform arithmetic yourself."
    )

    def __init__(self, *, llm: ChatOpenAI | None = None, server_url: str | None = None) -> None:
        super().__init__(tool_name="sub", prompt=self.PROMPT, llm=llm, server_url=server_url)


def _format_trace(result: str, trace: tuple[ReactionStep, ...]) -> str:
    if not trace:
        return f"Result: {result}"

    lines = [f"Result: {result}", "Steps:"]
    for idx, step in enumerate(trace, start=1):
        lines.append(
            f"  {idx}. thought={step.thought!r} action={step.action}"
            f" args={step.arguments} observation={step.observation}"
        )
    return "\n".join(lines)


def format_decimal(value: Decimal) -> str:
    normalized = value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


__all__ = [
    "MCPToolAgentExecutor",
    "AdditionAgentExecutor",
    "SubtractionAgentExecutor",
    "format_decimal",
]
