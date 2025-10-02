"""Convenience exports for A2A arithmetic components."""

from .remote_executors import (
    MCPToolAgentExecutor,
    AdditionAgentExecutor,
    SubtractionAgentExecutor,
    MultiplicationAgentExecutor,
    DivisionAgentExecutor,
    format_decimal,
)
from .clients import (
    AdditionClient,
    SubtractionClient,
    MultiplicationClient,
    DivisionClient,
)
from .host_executor import ArithmeticHostAgentExecutor

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
