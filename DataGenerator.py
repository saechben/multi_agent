"""Generate random arithmetic expressions and store them with their answers."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Iterable, Sequence

# Ensure we have enough precision for chained divisions/multiplications
getcontext().prec = 28

DATA_DIR = Path("data")


def format_decimal(value: Decimal) -> str:
    normalized = value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


class DataGenerator:
    """Generate random arithmetic expressions with deterministic answers."""

    def __init__(
        self,
        *,
        sample_count: int,
        operand_count: int,
        operators: Sequence[str],
        value_range: tuple[int, int] = (1, 100),
        seed: int | None = None,
    ) -> None:
        if operand_count < 2:
            raise ValueError("operand_count must be at least 2")
        unsupported = set(operators) - {"+", "-", "*", "/"}
        if unsupported:
            raise ValueError(f"Unsupported operators: {', '.join(sorted(unsupported))}")
        if not operators:
            raise ValueError("At least one operator must be provided")

        self.sample_count = sample_count
        self.operand_count = operand_count
        self.operators = list(operators)
        self.value_range = value_range
        self._rng = random.Random(seed)

    def iter_samples(self) -> Iterable[dict[str, str]]:
        low, high = self.value_range
        for _ in range(self.sample_count):
            operands = [self._rng.randint(low, high) for _ in range(self.operand_count)]
            ops = [self._rng.choice(self.operators) for _ in range(self.operand_count - 1)]
            expression = self._build_expression(operands, ops)
            result = self._compute_expression_result(operands, ops)
            yield {"expression": expression, "result": format_decimal(result)}

    def write(self, output_path: Path | None = None) -> Path:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if output_path is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            output_path = DATA_DIR / f"dataset-{timestamp}.jsonl"
        elif not output_path.is_absolute():
            output_path = DATA_DIR / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with output_path.open("w", encoding="utf-8") as fh:
            for item in self.iter_samples():
                fh.write(json.dumps(item) + "\n")
                count += 1

        print(f"Wrote {count} samples to {output_path}")
        return output_path

    @staticmethod
    def _build_expression(operands: Sequence[int], operators: Sequence[str]) -> str:
        parts: list[str] = [str(operands[0])]
        for operator, operand in zip(operators, operands[1:]):
            parts.append(operator)
            parts.append(str(operand))
        return " ".join(parts)

    @staticmethod
    def _compute_expression_result(operands: Sequence[int], operators: Sequence[str]) -> Decimal:
        if len(operands) != len(operators) + 1:
            raise ValueError("Operator count must be exactly one less than operand count")

        result = Decimal(operands[0])
        for operator, operand in zip(operators, operands[1:]):
            operand_value = Decimal(operand)
            if operator == "+":
                result += operand_value
            elif operator == "-":
                result -= operand_value
            elif operator == "*":
                result *= operand_value
            elif operator == "/":
                result /= operand_value
            else:
                raise ValueError(f"Unsupported operator '{operator}'")
        return result


def generate_dataset(
    *,
    samples: int,
    operands: int,
    operators: Sequence[str],
    seed: int | None = None,
    output: Path | None = None,
) -> Path:
    generator = DataGenerator(
        sample_count=samples,
        operand_count=operands,
        operators=operators,
        seed=seed,
    )
    return generator.write(output)


if __name__ == "__main__":
    output_path = Path("./asmd_easy.json")
    samples = 100
    operands = 3
    operators = ["+","-","*","/"]
    seed = 123
    generate_dataset(samples=samples,
                     operands=operands,
                     operators=operators,
                     seed=seed,
                     output=output_path)