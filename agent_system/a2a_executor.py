"""Backward-compatible re-exports for A2A arithmetic helpers."""

from .arithmetic_a2a import (
    MCPToolAgentExecutor,
    AdditionAgentExecutor,
    SubtractionAgentExecutor,
    MultiplicationAgentExecutor,
    DivisionAgentExecutor,
    AdditionClient,
    SubtractionClient,
    MultiplicationClient,
    DivisionClient,
    ArithmeticHostAgentExecutor,
    format_decimal,
)

__all__ = [
    "MCPToolAgentExecutor",
    "AdditionAgentExecutor",
    "SubtractionAgentExecutor",
    "MultiplicationAgentExecutor",
    "DivisionAgentExecutor",
    "AdditionClient",
    "SubtractionClient",
    "MultiplicationClient",
    "DivisionClient",
    "ArithmeticHostAgentExecutor",
    "format_decimal",
]
