"""Backward-compatible re-exports for A2A arithmetic helpers."""

from .arithmetic_a2a import (
    MCPToolAgentExecutor,
    AdditionAgentExecutor,
    SubtractionAgentExecutor,
    AdditionClient,
    SubtractionClient,
    ArithmeticHostAgentExecutor,
    format_decimal,
)

__all__ = [
    "MCPToolAgentExecutor",
    "AdditionAgentExecutor",
    "SubtractionAgentExecutor",
    "AdditionClient",
    "SubtractionClient",
    "ArithmeticHostAgentExecutor",
    "format_decimal",
]
