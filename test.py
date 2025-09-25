"""Quick sanity checks for the arithmetic MCP server."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Iterable, Sequence

from mcp.types import ContentBlock, TextContent

from agent_system.MCP import mcp


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
    asyncio.run(run_sample_operations())
