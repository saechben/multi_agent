"""Client helpers for interacting with the arithmetic host via A2A."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Sequence

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    JSONRPCErrorResponse,
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
    SendMessageSuccessResponse,
    Task,
    TextPart,
)
from httpx import Timeout


@dataclass
class HostExecution:
    """Outcome of a host agent invocation."""

    expression: str
    final_value: Decimal
    messages: tuple[str, ...]
    result: Task | Message

    @property
    def task_id(self) -> str | None:
        if isinstance(self.result, Task):
            return self.result.id
        if isinstance(self.result, Message):
            return self.result.task_id
        return None


class HostClient:
    """Convenience wrapper around the host agent's A2A endpoint."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._httpx_client = httpx.AsyncClient()
        self._resolver = A2ACardResolver(httpx_client=self._httpx_client, base_url=self._base_url)
        self._client: A2AClient | None = None

    async def _ensure_client(self) -> None:
        if self._client is not None:
            return
        agent_card = await self._resolver.get_agent_card()
        self._client = A2AClient(httpx_client=self._httpx_client, agent_card=agent_card)

    async def evaluate(self, expression: str) -> HostExecution:
        await self._ensure_client()
        assert self._client is not None
        message = Message(
            role=Role.user,
            messageId=str(uuid.uuid4()),
            parts=[Part(root=TextPart(text=expression))],
        )
        request = SendMessageRequest(
            id=str(uuid.uuid4()),
            params=MessageSendParams(message=message),
        )
        response = await self._client.send_message(request, http_kwargs={"timeout": Timeout(None)})
        payload = response.root
        if isinstance(payload, JSONRPCErrorResponse):
            raise RuntimeError(f"Host agent error: {payload.error.message}")
        assert isinstance(payload, SendMessageSuccessResponse)
        result = payload.result
        messages = self._collect_messages(result)
        final_text = self._extract_final_text(result, messages)
        final_value = self._parse_final_value(final_text, messages)
        return HostExecution(
            expression=expression,
            final_value=final_value,
            messages=tuple(messages),
            result=result,
        )

    async def aclose(self) -> None:
        await self._httpx_client.aclose()

    def _collect_messages(self, result: Task | Message) -> list[str]:
        outputs: list[str] = []
        if isinstance(result, Task):
            if result.history:
                for entry in result.history:
                    if entry.role != Role.agent:
                        continue
                    text = self._message_to_text(entry)
                    if text and (not outputs or outputs[-1] != text):
                        outputs.append(text)
            if result.status and result.status.message:
                text = self._message_to_text(result.status.message)
                if text and (not outputs or outputs[-1] != text):
                    outputs.append(text)
        else:
            if result.role == Role.agent:
                outputs.append(self._message_to_text(result))
        return outputs

    @staticmethod
    def _message_to_text(message: Message) -> str:
        parts: list[str] = []
        for part in message.parts:
            root = part.root
            if isinstance(root, TextPart):
                parts.append(root.text)
        return "\n".join(parts)

    @staticmethod
    def _extract_final_text(result: Task | Message, messages: Sequence[str]) -> str:
        if isinstance(result, Task):
            status_message = getattr(result, "status", None)
            if status_message and status_message.message:
                text = HostClient._message_to_text(status_message.message)
                if text:
                    return text
        else:
            text = HostClient._message_to_text(result)
            if text:
                return text
        if messages:
            return messages[-1]
        raise RuntimeError("Host agent response did not include any text output")

    @staticmethod
    def _parse_final_value(final_text: str, messages: Sequence[str]) -> Decimal:
        numeric_pattern = re.compile(r"[-+]?\d+(?:\.\d+)?")

        try:
            return Decimal(final_text.strip())
        except InvalidOperation:
            pass

        matches = numeric_pattern.findall(final_text)
        for value in reversed(matches):
            try:
                return Decimal(value)
            except InvalidOperation:
                continue

        for text in reversed(messages):
            stripped = text.strip()
            if stripped:
                try:
                    return Decimal(stripped)
                except InvalidOperation:
                    pass

            matches = numeric_pattern.findall(text)
            for value in reversed(matches):
                try:
                    return Decimal(value)
                except InvalidOperation:
                    continue

        raise RuntimeError("Unable to determine numeric result from agent messages")


__all__ = ["HostClient", "HostExecution"]
