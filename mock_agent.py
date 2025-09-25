"""Mock agent that exercises the arithmetic MCP server over Streamable HTTP."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Iterable

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult, TextContent

SERVER_URL = "http://127.0.0.1:8000/mcp"


def extract_decimal(result: CallToolResult) -> Decimal:
    """Convert the tool result payload into a Decimal."""

    if result.structuredContent and "result" in result.structuredContent:
        return Decimal(str(result.structuredContent["result"]))

    for block in result.content:
        if isinstance(block, TextContent):
            return Decimal(block.text)

    raise RuntimeError("Tool response did not contain numeric content")


async def run_mock_agent() -> None:
    async with streamablehttp_client(SERVER_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            init = await session.initialize()
            server = init.serverInfo
            print(f"Connected to MCP server: {server.name} v{server.version}")

            tools = await session.list_tools()
            tool_names = ", ".join(tool.name for tool in tools.tools)
            print("Available tools:", tool_names)

            cases: Iterable[tuple[str, list[str]]] = (
                ("add", ["2", "3.5", "4.5"]),
                ("sub", ["10", "1.5", "2.5"]),
                ("mul", ["1.5", "4", "2"]),
                ("div", ["20", "2", "2"]),
            )

            for operation, operands in cases:
                result = await session.call_tool(operation, {"operands": operands})
                if result.isError:
                    print(f"{operation} failed: {result}")
                    continue

                value = extract_decimal(result)
                joined = ", ".join(operands)
                print(f"{operation}({joined}) -> {value}")


def main() -> None:
    asyncio.run(run_mock_agent())


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
