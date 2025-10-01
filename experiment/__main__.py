"""CLI wrapper for running a single arithmetic experiment expression."""

from __future__ import annotations

import argparse
from dotenv import load_dotenv
from .manager import run


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run an arithmetic host experiment")
    parser.add_argument("expression", help="Expression to send to the host agent")
    args = parser.parse_args()
    outcome = run(args.expression)
    print("Expression:", outcome.expression)
    print("Final value:", outcome.final_value)
    print("Agent messages:")
    for line in outcome.messages:
        print("-", line)
    if outcome.task_id:
        print("Task ID:", outcome.task_id)


if __name__ == "__main__":
    main()
