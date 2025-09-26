"""Example script that drives the addition MCP tool via a LangGraph ReAct agent."""

from __future__ import annotations

import asyncio
import os
from typing import Sequence

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from agent_system.Agent import LangGraphReactAgent, ReactionStep


from decimal import Decimal
from typing import Iterable, Sequence

from mcp.types import ContentBlock, TextContent

from agent_system.MCP import mcp

load_dotenv()
PROMPT = (
    "You are an addition specialist. You must always call the 'add' MCP tool "
    "to compute totals and never perform arithmetic yourself. After receiving a tool "
    "observation, report the numeric result to the user."
)


async def run_addition_agent(expression: str) -> tuple[str, Sequence[ReactionStep]]:
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_API_KEY environment variable is not set")

    model_id = os.getenv("LLM_MODEL", "gpt-4o-mini")
    server_url = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")

    llm = ChatOpenAI(model=model_id, api_key=api_key, temperature=0.0)

    agent = LangGraphReactAgent(
        name="adder-agent",
        server_url=server_url,
        llm=llm,
        prompt=PROMPT,
        tool_names=["add"],
    )

    await agent.initialize()
    try:
        result = await agent.handle_task({"description": expression})
        if any(step.action not in {None, "add"} for step in agent.trace):
            raise RuntimeError("Agent invoked a tool other than 'add'")
        return result, agent.trace
    finally:
        await agent.shutdown()


async def main() -> None:
    expression = os.getenv("ADDITION_PROMPT", "Add 17 and 29")
    result, trace = await run_addition_agent(expression)
    print(f"Input: {expression}")
    print(f"Result: {result}")
    print("Trace:")
    for idx, step in enumerate(trace, 1):
        print(f"  Step {idx}: thought={step.thought!r}, action={step.action}, arguments={step.arguments}, observation={step.observation}")



"""Quick sanity checks for the arithmetic MCP server."""



def _extract_decimal(blocks: Sequence[ContentBlock]) -> Decimal:
    texts = [block.text for block in blocks if isinstance(block, TextContent)]
    if not texts:
        raise RuntimeError("Tool response did not include textual content")
    return Decimal(texts[0])


async def run_sample_operations() -> None:
    cases: Iterable[tuple[str, list[str], Decimal]] = (
        ("add", ["2", "3.5", "4.5"], Decimal("10")),
        ("sub", ["10", "1.5", "2.5"], Decimal("6")),
        ("mul", ["1.5", "4", "2"], Decimal("12")),
        ("div", ["20", "2", "2"], Decimal("5")),
    )

    for operation, operands, expected in cases:
        response = await mcp.call_tool(operation, {"operands": operands})
        if isinstance(response, tuple):
            blocks, structured = response
            if isinstance(structured, dict) and "result" in structured:
                value = Decimal(str(structured["result"]))
            else:
                value = _extract_decimal(blocks)
        else:
            blocks = response
            value = _extract_decimal(blocks)

        assert value == expected, f"{operation} expected {expected} but received {value}"
        print(f"{operation}({', '.join(operands)}) = {value}")

    tools = await mcp.list_tools()
    tool_names = [tool.name for tool in tools]
    print("Available MCP tools:", ", ".join(tool_names))

if __name__ == "__main__":
    asyncio.run(main())


