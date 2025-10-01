"""Configuration objects for experiment orchestration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Settings for launching the arithmetic experiment cluster."""

    host: str = "127.0.0.1"
    mcp_port: int = 18200
    addition_port: int = 18201
    subtraction_port: int = 18202
    host_port: int = 18203
    log_root: Path = Path("logs")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_api_key: str | None = None


__all__ = ["ExperimentConfig"]
