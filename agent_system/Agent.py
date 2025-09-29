"""Agent abstractions and LangGraph-powered ReAct agent using MCP tools."""

from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import create_react_agent
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult, Tool as MCPTool, TextContent
from pydantic import BaseModel, Field, create_model
from abc import ABC

class AgentInterface(ABC):
    """Common behaviour every agent implementation must provide."""

    name: str

    async def initialize(self) -> None:  
        raise NotImplementedError

    async def handle_task(self, task: Mapping[str, Any]) -> str:  
        raise NotImplementedError

    async def shutdown(self) -> None:  
        raise NotImplementedError


@dataclass
class ReactionStep:
    """Single reasoning/action step captured from the agent trace."""

    thought: str
    action: str | None
    arguments: Mapping[str, Any] | None
    observation: str | None = None


class AgentState(TypedDict):
    """Minimal view of agent execution for external inspection."""

    messages: Sequence[BaseMessage]
    steps: Sequence[ReactionStep]
    answer: str


@dataclass
class LangGraphReactAgent(AgentInterface):
    """ReAct agent template backed by LangGraph's prebuilt factory and MCP tools."""

    name: str
    server_url: str
    llm: Any
    prompt: str | BaseMessage | None = None
    tool_names: Sequence[str] | None = None
    headers: Mapping[str, str] | None = None
    timeout: float = 30.0
    sse_read_timeout: float = 300.0

    _stack: AsyncExitStack | None = field(init=False, default=None)
    _session: ClientSession | None = field(init=False, default=None)
    _graph: Any = field(init=False, default=None)
    _tools: list[BaseTool] = field(init=False, default_factory=list)
    _initialized: bool = field(init=False, default=False)
    _trace: list[ReactionStep] = field(init=False, default_factory=list)
    _last_messages: Sequence[BaseMessage] | None = field(init=False, default=None)

    @property
    def trace(self) -> Sequence[ReactionStep]:
        return tuple(self._trace)

    async def initialize(self) -> None:
        if self._initialized:
            return

        stack = AsyncExitStack()
        transport = streamablehttp_client(
            url=self.server_url,
            headers=dict(self.headers or {}),
            timeout=self.timeout,
            sse_read_timeout=self.sse_read_timeout,
        )
        read_stream, write_stream, _ = await stack.enter_async_context(transport)
        session = ClientSession(read_stream, write_stream)
        session = await stack.enter_async_context(session)
        await session.initialize()

        tools_response = await session.list_tools()
        mcp_tools = self._select_tools(tools_response.tools)
        if not mcp_tools:
            await stack.aclose()
            raise RuntimeError("No MCP tools available for LangGraph agent")

        self._tools = [self._build_tool(tool) for tool in mcp_tools]
        self._graph = create_react_agent(
            model=self.llm,
            tools=self._tools,
            prompt=self.prompt,
            name=f"{self.name}-react",
        )

        self._stack = stack
        self._session = session
        self._initialized = True

    async def handle_task(self, task: Mapping[str, Any]) -> str:
        await self.initialize()
        if self._graph is None:
            raise RuntimeError("Agent graph is not initialized")

        payload = self._build_input_messages(task)
        try:
            result = await self._graph.ainvoke({"messages": payload})
        except Exception as exc:
            raise RuntimeError(f"LangGraph execution failed for payload {payload}") from exc
        messages = result.get("messages", [])
        self._last_messages = messages
        self._trace = self._build_trace(messages)
        answer = self._extract_final_answer(messages)
        return answer

    async def shutdown(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None
        self._graph = None
        self._tools = []
        self._initialized = False
        self._last_messages = None
        self._trace = []

    # ------------------------------------------------------------------
    # Tool handling helpers
    # ------------------------------------------------------------------
    def _select_tools(self, tools: Sequence[MCPTool]) -> list[MCPTool]:
        if not self.tool_names:
            return list(tools)
        allowed = set(self.tool_names)
        return [tool for tool in tools if tool.name in allowed]
    #so Langchain agent can use tools natively 
    def _build_tool(self, tool: MCPTool) -> StructuredTool:
        args_schema = self._create_pyd_schema(tool)

        async def _runner(**kwargs: Any) -> str:
            return await self._call_tool(tool.name, kwargs)

        description = tool.description or f"Invoke MCP tool '{tool.name}'."
        return StructuredTool(
            name=tool.name,
            description=description,
            args_schema=args_schema,
            coroutine=_runner,
        )
    #Langchain structuredTool expects a pydantic model that describes the tool's arguments, but MCP exposes as JSON schema for tool inputs --> this method does the conversion
    def _create_pyd_schema(self, tool: MCPTool) -> type[BaseModel]:
        schema = tool.inputSchema or {}
        properties = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        fields: dict[str, tuple[type[Any], Field]] = {}

        for prop_name, prop_schema in properties.items():
            annotation: type[Any] = Any
            prop_type = prop_schema.get("type")
            if prop_type == "string":
                annotation = str
            elif prop_type == "number":
                annotation = float
            elif prop_type == "integer":
                annotation = int
            elif prop_type == "array":
                annotation = list[Any]
            default = ... if prop_name in required else None
            field = Field(
                default,
                description=prop_schema.get("description"),
                title=prop_schema.get("title"),
            )
            fields[prop_name] = (annotation, field)

        if not fields:
            fields["payload"] = (dict[str, Any], Field(..., description="Raw MCP tool arguments"))

        model_name = f"{tool.name.title()}Args"
        return create_model(model_name, **fields)  

    async def _call_tool(self, tool_name: str, arguments: Mapping[str, Any]) -> str:
        if self._session is None:
            raise RuntimeError("Agent session is not initialized")
        result = await self._session.call_tool(tool_name, dict(arguments))
        return self._parse_tool_result(result)

    @staticmethod
    def _parse_tool_result(result: CallToolResult) -> str:
        if result.isError:
            raise RuntimeError(f"Tool returned error payload: {result.structuredContent}")
        if result.structuredContent and "result" in result.structuredContent:
            return str(result.structuredContent["result"])
        for block in result.content:
            if isinstance(block, TextContent):
                return block.text
        raise RuntimeError("Tool response did not include readable content")

    # ------------------------------------------------------------------
    # LangGraph output processing
    # ------------------------------------------------------------------
    def _build_input_messages(self, task: Mapping[str, Any]) -> list[dict[str, Any]]:
        if "messages" in task:
            messages = task["messages"]
            if isinstance(messages, list):
                return list(messages)
            raise TypeError("Task 'messages' must be a list of message dicts")

        description = (
            task.get("description")
            or task.get("query")
            or task.get("problem")
            or task.get("prompt")
            or ""
        )
        return [{"role": "user", "content": str(description)}]

    def _build_trace(self, messages: Sequence[BaseMessage]) -> list[ReactionStep]:
        trace: list[ReactionStep] = []
        steps_by_call_id: dict[str, ReactionStep] = {}

        for message in messages:
            if isinstance(message, AIMessage):
                thought = self._message_to_text(message)
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        step = ReactionStep(
                            thought=thought,
                            action=tool_call.get("name"),
                            arguments=tool_call.get("args"),
                        )
                        call_id = tool_call.get("id")
                        if call_id:
                            steps_by_call_id[call_id] = step
                        trace.append(step)
                elif trace:
                    if trace[-1].observation is None:
                        trace[-1].observation = thought
            elif isinstance(message, ToolMessage):
                step = steps_by_call_id.get(message.tool_call_id)
                if step is not None:
                    step.observation = self._content_to_text(message.content)

        return trace

    def _extract_final_answer(self, messages: Sequence[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, AIMessage) and not message.tool_calls:
                return self._message_to_text(message)
        if messages:
            return self._content_to_text(messages[-1].content)
        raise RuntimeError("Agent did not return any messages")

    def _message_to_text(self, message: AIMessage) -> str:
        return self._content_to_text(message.content)

    def _content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for chunk in content:
                if isinstance(chunk, str):
                    parts.append(chunk)
                elif isinstance(chunk, dict):
                    value = chunk.get("text") or chunk.get("content")
                    if value is not None:
                        parts.append(str(value))
                else:
                    parts.append(str(chunk))
            return "\n".join(parts)
        return str(content)


__all__ = [
    "AgentInterface",
    "ReactionStep",
    "AgentState",
    "LangGraphReactAgent",
]
