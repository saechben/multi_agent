"""Evaluate the host agent on a generated dataset of arithmetic expressions."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable

from experiment import Experiment, ExperimentConfig

RESULTS_DIR = Path("results")


def load_dataset(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    samples: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def parse_decimal(text: str) -> Decimal:
    try:
        return Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid decimal value '{text}'") from exc


class EvaluationPipeline:
    """Run the host agent against a dataset and compute accuracy."""

    def __init__(
        self,
        dataset_path: Path,
        *,
        config: ExperimentConfig | None = None,
        results_dir: Path = RESULTS_DIR,
    ) -> None:
        self.dataset_path = dataset_path
        self.config = config or ExperimentConfig()
        self.results_dir = results_dir

    async def _evaluate_async(self) -> dict[str, object]:
        samples = load_dataset(self.dataset_path)
        if not samples:
            raise ValueError("Dataset is empty")

        correct = 0
        total = len(samples)
        mismatches: list[dict[str, object]] = []

        async with Experiment(self.config) as experiment:
            for sample in samples:
                expression = sample["expression"]
                expected = parse_decimal(sample["result"])
                outcome = await experiment.evaluate(expression)
                actual = outcome.final_value

                if actual == expected:
                    correct += 1
                else:
                    mismatches.append(
                        {
                            "expression": expression,
                            "expected": str(expected),
                            "actual": str(actual),
                            "messages": list(outcome.messages),
                            "task_id": outcome.task_id,
                        }
                    )

        accuracy = correct / total
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        return {
            "timestamp": timestamp,
            "dataset": str(self.dataset_path),
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "mismatches": mismatches,
        }

    def run(self) -> Path:
        """Execute the evaluation and persist the results."""
        results = asyncio.run(self._evaluate_async())
        self.results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        output_path = self.results_dir / f"evaluation-{timestamp}.json"
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        return output_path


def run_evaluation(dataset_path: Path, *, config: ExperimentConfig | None = None, results_dir: Path = RESULTS_DIR) -> Path:
    pipeline = EvaluationPipeline(dataset_path, config=config, results_dir=results_dir)
    return pipeline.run()



