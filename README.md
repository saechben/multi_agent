Context for Multi‑Agent A2A Arithmetic System

Goal

The objective of this project is to build a multi‑agent system that evaluates the A2A (agent‑to‑agent) communication protocol by solving simple arithmetic expressions, such as 2*(3+4/3).  Arithmetic tasks are chosen because they have well‑defined ground‑truth outputs, making it easy to verify correctness.  By measuring how accurately and efficiently the agents solve these expressions, we can analyse the coordination costs, limitations and effectiveness of the A2A protocol.

General Approach

Shared MCP Server

To keep the system simple and representative of real‑world deployments, a single Model Context Protocol (MCP) server is used to provide all arithmetic operations.  This shared server exposes operations such as addition, subtraction, multiplication and division.  Multiple agents can call the same server concurrently.  Using one server avoids version drift between tools and simplifies monitoring, while still allowing for later experiments with fault‑injection or per‑agent isolation.

Agents and Architecture
	•	Planner agent – Parses an input expression into an abstract syntax tree (AST), plans the sequence of operations needed to evaluate the expression and delegates tasks to the appropriate worker agents.
	•	Worker agents – Each worker agent is responsible for a single arithmetic operation (e.g., add, sub, mul, div).  A worker agent accepts a tool_call message, invokes the MCP server to perform its operation on the provided operands and returns the result.  Every agent runs in its own Docker container and exposes a /act endpoint via FastAPI for communication.
	•	Broker/orchestrator (optional) – A central component that routes messages among agents, enforces step budgets and deadlines, and collects logs and metrics.  It can also compare final outputs against a trusted evaluator to determine correctness.  Including a broker is useful for evaluating system behaviour under controlled budgets and failure scenarios but is not mandatory for a basic implementation.

Communication and Protocol

Agents communicate via the A2A protocol using structured JSON messages.  Each message includes metadata (a unique trace_id, current step number, sender and receiver IDs) and an intent (e.g., tool_call, delegate, finalize).  The content contains the operation to perform and its arguments.  The planner sends tool_call messages to worker agents specifying the operation and operands.  Worker agents call the MCP server with the requested operation and return the result to the planner.  All messages are logged with timestamps and other context to facilitate analysis of communication patterns and performance.

Evaluation Methodology

To assess the A2A protocol’s performance and reliability, the system will collect and analyse various metrics:
	•	Accuracy – Compare each agent‑generated result with a trusted evaluator implemented using deterministic numeric types (e.g., Python decimal.Decimal or fractions.Fraction with fixed precision).  Count correct vs. incorrect computations.
	•	Latency and step count – Measure end‑to‑end latency for each expression, as well as the number of messages exchanged and tool calls made.  Use these metrics to quantify coordination cost.
	•	Error analysis – Categorise failures (planning errors, wrong tool usage, numeric rounding or overflow issues, network/protocol errors) using logs and message metadata.
	•	Curriculum of expressions – Evaluate the system on a range of expressions varying in depth and complexity.  Include edge cases such as division by zero, nested parentheses, negative numbers and very large or very small values.
	•	Instrumentation – Export logs and metrics (e.g., via Prometheus) from the broker and agents.  Record per‑message timestamps and tool call latencies.  Analyse this data to derive insights into the cost of coordination and identify bottlenecks.

Deterministic Operations

To ensure that numeric errors do not confound the evaluation of the A2A protocol, all arithmetic operations on the MCP server should be pure and deterministic.  Implement operations using types like decimal.Decimal or fractions.Fraction with fixed precision, returning exact or high‑precision results and avoiding floating‑point drift.


