"""Host agent that decomposes arithmetic expressions and delegates to remote agents."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import List

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.utils import new_agent_text_message
from a2a.types import TaskState, TaskStatus, TaskStatusUpdateEvent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

from .clients import AdditionClient, SubtractionClient, MultiplicationClient, DivisionClient
from .remote_executors import format_decimal


class DecompositionStep(BaseModel):
    operation: str = Field(pattern="^(add|sub|mul|div)$")
    operand: Decimal


class DecompositionResult(BaseModel):
    initial_value: Decimal
    steps: List[DecompositionStep]


@dataclass
class ArithmeticHostAgentExecutor(AgentExecutor):
    """Orchestrates remote addition and subtraction agents using LLM planning."""

    planner_llm: ChatOpenAI | None = None
    addition_client: AdditionClient | None = None
    subtraction_client: SubtractionClient | None = None
    multiplication_client: MultiplicationClient | None = None
    division_client: DivisionClient | None = None

    def __post_init__(self) -> None:
        if self.planner_llm is None:
            api_key = os.getenv("LLM_API_KEY")
            if not api_key:
                raise RuntimeError("LLM_API_KEY must be set for the host agent")
            model_id = os.getenv("LLM_MODEL", "gpt-4o-mini")
            self.planner_llm = ChatOpenAI(model=model_id, api_key=api_key, temperature=0.0)
        if self.addition_client is None:
            self.addition_client = AdditionClient(os.getenv("ADDITION_AGENT_URL", "http://localhost:9999"))
        if self.subtraction_client is None:
            self.subtraction_client = SubtractionClient(os.getenv("SUBTRACTION_AGENT_URL", "http://localhost:10000"))
        if self.multiplication_client is None:
            self.multiplication_client = MultiplicationClient(os.getenv("MULTIPLICATION_AGENT_URL", "http://localhost:10001"))
        if self.division_client is None:
            self.division_client = DivisionClient(os.getenv("DIVISION_AGENT_URL", "http://localhost:10002"))

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        expression = (context.get_user_input() or "").strip()
        if not expression:
            await event_queue.enqueue_event(
                new_agent_text_message("Please provide an arithmetic expression to evaluate."),
            )
            return

        task_id = context.task_id or (context.current_task.id if context.current_task else None)
        context_id = context.context_id or (context.current_task.context_id if context.current_task else None)

        async def send_status(text: str, *, state: TaskState = TaskState.working, final: bool = False) -> None:
            if not task_id or not context_id:
                await event_queue.enqueue_event(new_agent_text_message(text))
                return
            message = new_agent_text_message(text)
            message.task_id = task_id
            message.context_id = context_id
            status = TaskStatus(state=state, message=message)
            event = TaskStatusUpdateEvent(
                taskId=task_id,
                contextId=context_id,
                status=status,
                final=final,
            )
            await event_queue.enqueue_event(event)

        await send_status(f"Planning expression: {expression}")

        try:
            plan = await self._decompose_with_llm(expression)
        except Exception as exc:
            await event_queue.enqueue_event(
                new_agent_text_message(f"Unable to decompose expression: {exc}"),
            )
            return

        plan_lines = [
            "Plan:",
            f"  start = {format_decimal(plan.initial_value)}",
        ]
        symbol_map = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
        for idx, step in enumerate(plan.steps, start=1):
            operator = symbol_map.get(step.operation, step.operation)
            plan_lines.append(
                f"  {idx}. {operator} {format_decimal(step.operand)}",
            )
        await send_status("\n".join(plan_lines))

        current = plan.initial_value
        for idx, step in enumerate(plan.steps, start=1):
            operator = symbol_map.get(step.operation, step.operation)
            if step.operation == "add":
                result = await self.addition_client.add(current, step.operand)  # type: ignore[arg-type]
            elif step.operation == "sub":
                result = await self.subtraction_client.subtract(current, step.operand)  # type: ignore[arg-type]
            elif step.operation == "mul":
                result = await self.multiplication_client.multiply(current, step.operand)  # type: ignore[arg-type]
            elif step.operation == "div":
                result = await self.division_client.divide(current, step.operand)  # type: ignore[arg-type]
            else:  # pragma: no cover - guard against unexpected operations
                await send_status(f"Unsupported operation '{step.operation}'", state=TaskState.failed, final=True)
                return

            await send_status(
                f"Step {idx}: {format_decimal(current)} {operator} {format_decimal(step.operand)} = {format_decimal(result)}"
            )
            current = result

        await send_status(
            format_decimal(current),
            state=TaskState.completed,
            final=True,
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(new_agent_text_message("Cancellation is not supported."))

    async def shutdown(self) -> None:
        await self.addition_client.shutdown()  # type: ignore[union-attr]
        await self.subtraction_client.shutdown()  # type: ignore[union-attr]
        await self.multiplication_client.shutdown()  # type: ignore[union-attr]
        await self.division_client.shutdown()  # type: ignore[union-attr]

    async def _decompose_with_llm(self, expression: str) -> DecompositionResult:
        prompt = (
            "You are an expert math planner. Break down the following arithmetic expression into"
            " a sequence of operations that only use addition, subtraction, multiplication, or division."
            " Output JSON matching this schema: {\n"
            "  \"initial_value\": number,\n"
            "  \"steps\": [{\"operation\": \"add\" | \"sub\" | \"mul\" | \"div\", \"operand\": number}]\n"
            "}.\n"
            "Expression: "
            + expression
        )
        response = await self.planner_llm.ainvoke(prompt)
        content = response.content if isinstance(response.content, str) else json.dumps(response.content)

        try:
            data = json.loads(content)
            plan = DecompositionResult.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            plan = self._fallback_parse(expression)
            if plan is None:
                raise RuntimeError(f"Invalid decomposition response: {exc}") from exc
        return plan

    def _fallback_parse(self, expression: str) -> DecompositionResult | None:
        try:
            tokens = self._tokenize_expression(expression)
        except ValueError:
            return None

        first = tokens[0]
        if not isinstance(first, Decimal):
            return None
        steps: list[DecompositionStep] = []
        idx = 1
        while idx < len(tokens):
            operator = tokens[idx]
            operand = tokens[idx + 1]
            if not isinstance(operator, str) or not isinstance(operand, Decimal):
                return None
            if operator == "+":
                op_key = "add"
            elif operator == "-":
                op_key = "sub"
            elif operator == "*":
                op_key = "mul"
            elif operator == "/":
                op_key = "div"
            else:
                return None
            steps.append(
                DecompositionStep(
                    operation=op_key,
                    operand=operand,
                )
            )
            idx += 2
        return DecompositionResult(initial_value=first, steps=steps)

    def _tokenize_expression(self, expression: str) -> list[Decimal | str]:
        tokens: list[Decimal | str] = []
        i = 0
        length = len(expression)
        expect_number = True
        current_sign = 1
        while i < length:
            ch = expression[i]
            if ch.isspace():
                i += 1
                continue
            if expect_number:
                if ch in "+-" and not tokens:
                    current_sign = 1 if ch == "+" else -1
                    i += 1
                    continue
                start = i
                while i < length and (expression[i].isdigit() or expression[i] == "."):
                    i += 1
                if start == i:
                    raise ValueError("Expected number")
                try:
                    value = Decimal(expression[start:i])
                except InvalidOperation as exc:
                    raise ValueError("Invalid number") from exc
                tokens.append(value * current_sign)
                current_sign = 1
                expect_number = False
            else:
                if ch not in "+-*/":
                    raise ValueError("Expected operator")
                tokens.append(ch)
                expect_number = True
                i += 1
        if expect_number:
            raise ValueError("Expression cannot end with an operator")
        return tokens


__all__ = ["ArithmeticHostAgentExecutor", "DecompositionResult", "DecompositionStep"]
