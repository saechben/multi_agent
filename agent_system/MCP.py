"""Arithmetic MCP server implemented with the Model Context Protocol SDK."""

from __future__ import annotations

from decimal import Decimal, DivisionByZero, getcontext
from typing import Iterable

from mcp.server.fastmcp import FastMCP


DEFAULT_PRECISION = 28
getcontext().prec = DEFAULT_PRECISION

mcp = FastMCP(
    name="arithmetic-mcp",
    instructions=(
        "Deterministic decimal arithmetic operations supporting addition, "
        "subtraction, multiplication, and division."
    ),
)


def _ensure_operands(operands: Iterable[Decimal]) -> list[Decimal]:
    values = list(operands)
    if len(values) < 2:
        raise ValueError("At least two operands are required")
    return values


@mcp.tool(name="add", description="Does addition")
def add(operands: list[Decimal]) -> str:
    values = _ensure_operands(operands)
    total = sum(values, Decimal(0))
    return str(total)


@mcp.tool(name="sub", description="Does subtraction")
def subtract(operands: list[Decimal]) -> str:
    values = _ensure_operands(operands)
    total = values[0]
    for value in values[1:]:
        total -= value
    return str(total)


@mcp.tool(name="mul", description="Does multiplication")
def multiply(operands: list[Decimal]) -> str:
    values = _ensure_operands(operands)
    product = values[0]
    for value in values[1:]:
        product *= value
    return str(product)


@mcp.tool(name="div", description="Divide the first operand by each subsequent operand in order.")
def divide(operands: list[Decimal]) -> str:
    values = _ensure_operands(operands)
    quotient = values[0]
    for value in values[1:]:
        try:
            quotient /= value
        except DivisionByZero as exc:
            raise ValueError("Division by zero is not allowed") from exc
    return str(quotient)


__all__ = ["mcp"]


def main() -> None:
    """Start the FastMCP server using the Streamable HTTP transport."""

    mcp.run("streamable-http")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
