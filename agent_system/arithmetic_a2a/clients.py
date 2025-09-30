"""A2A client helpers for interacting with remote arithmetic agents."""

from __future__ import annotations

import re
import uuid
from decimal import Decimal, InvalidOperation

import httpx
from httpx import Timeout
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    JSONRPCErrorResponse,
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
    SendMessageSuccessResponse,
    TextPart,
    Task,
)

from .remote_executors import format_decimal


class _A2AArithmeticClient:
    """Base class that handles A2A messaging and numeric response parsing."""

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

    async def send_prompt(self, prompt: str) -> Decimal:
        await self._ensure_client()
        assert self._client is not None
        message = Message(
            role=Role.user,
            messageId=str(uuid.uuid4()),
            parts=[Part(root=TextPart(text=prompt))],
        )
        request = SendMessageRequest(
            id=str(uuid.uuid4()),
            params=MessageSendParams(message=message),
        )
        response = await self._client.send_message(
            request,
            http_kwargs={"timeout": Timeout(None)},
        )
        payload = response.root
        if isinstance(payload, JSONRPCErrorResponse):
            raise RuntimeError(f"Remote agent error: {payload.error.message}")
        assert isinstance(payload, SendMessageSuccessResponse)
        return self._extract_decimal(payload.result)

    async def shutdown(self) -> None:
        await self._httpx_client.aclose()

    def _extract_decimal(self, result: Task | Message) -> Decimal:
        message: Message | None = None
        if isinstance(result, Task):
            if result.history:
                for item in reversed(result.history):
                    if item.role == Role.agent:
                        message = item
                        break
            if message is None and result.status and result.status.message:
                message = result.status.message
        else:
            message = result

        if message is None:
            raise RuntimeError("Agent response did not include a message")

        text = self._message_to_text(message)
        match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        if not match:
            raise RuntimeError(f"Unable to parse numeric result from '{text}'")
        try:
            return Decimal(match.group(0))
        except InvalidOperation as exc:  # pragma: no cover
            raise RuntimeError(f"Invalid numeric result: {match.group(0)}") from exc

    @staticmethod
    def _message_to_text(message: Message) -> str:
        parts: list[str] = []
        for part in message.parts:
            root = part.root
            if isinstance(root, TextPart):
                parts.append(root.text)
        return "\n".join(parts)


class AdditionClient(_A2AArithmeticClient):
    async def add(self, lhs: Decimal, rhs: Decimal) -> Decimal:
        prompt = f"Add {format_decimal(lhs)} and {format_decimal(rhs)}"
        return await self.send_prompt(prompt)


class SubtractionClient(_A2AArithmeticClient):
    async def subtract(self, lhs: Decimal, rhs: Decimal) -> Decimal:
        prompt = f"Subtract {format_decimal(rhs)} from {format_decimal(lhs)}"
        return await self.send_prompt(prompt)


__all__ = ["AdditionClient", "SubtractionClient"]
