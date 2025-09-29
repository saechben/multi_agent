"""Convenience exports for A2A arithmetic components."""

from .remote_executors import (
    MCPToolAgentExecutor,
    AdditionAgentExecutor,
    SubtractionAgentExecutor,
    format_decimal,
)
from .clients import AdditionClient, SubtractionClient
from .host_executor import ArithmeticHostAgentExecutor

__all__ = [
    "MCPToolAgentExecutor",
    "AdditionAgentExecutor",
    "SubtractionAgentExecutor",
    "AdditionClient",
    "SubtractionClient",
    "ArithmeticHostAgentExecutor",
    "format_decimal",
]
