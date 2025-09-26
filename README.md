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

Future Extensions

Although this context assumes a single MCP server, the architecture can be extended for more sophisticated experiments.  One might spin up per‑agent MCP servers to inject latency, simulate partial failures, or use different precision settings and then measure the impact on coordination.  Additional worker agents (e.g., for exponentiation or square roots) can also be introduced.  The modular design with containerised agents and a shared protocol enables flexible experimentation with the A2A paradigm.

A2A Protocol Alignment

The Agent2Agent (A2A) Protocol is an open standard developed by Google and donated to the Linux Foundation to enable seamless communication and collaboration between AI agents ￼.  It provides a common language that allows agents built using diverse frameworks or vendors to interoperate securely without exposing their internal logic, breaking down silos and fostering interoperability.

Core Concepts and Lifecycle
	•	Actors: A2A defines three roles: a User, a Client Agent (which formulates and sends tasks on behalf of the user), and a Remote Agent (which receives tasks and returns results) ￼.
	•	Agent Discovery: Agents publish an Agent Card, a standardized JSON document hosted at /.well-known/agent.json, containing metadata such as the agent’s name, version, hosting URL, description, supported input/output modalities, authentication methods, and advertised skills ￼.  Client agents use these cards to discover appropriate remote agents.
	•	Tasks: The primary communication object is a Task, representing an atomic unit of work that transitions through states such as submitted, working, input‑required and completed ￼.  A task contains messages for conversational exchanges and artifacts for immutable results; both are composed of parts (text, files, JSON, etc.) ￼.
	•	Workflow: A typical A2A workflow automates several steps ￼:
	1.	The user initiates a task with the client agent.
	2.	The client agent discovers a remote agent via its Agent Card.
	3.	The client agent sends a task/send request to the remote agent (JSON‑RPC over HTTP) containing a Task ID, an optional Session ID and an initial message.
	4.	The remote agent processes the task and responds with artifacts, intermediate messages or requests for additional input.
	5.	For long‑running tasks, the remote agent can stream partial results back to the client using Server‑Sent Events (SSE) ￼.
	6.	Tasks transition through states (submitted → working → input‑required → completed), and the client agent may resume or finalise the task accordingly.

A2A vs MCP

A2A and the Model Context Protocol (MCP) solve complementary challenges.  MCP focuses on agent‑to‑tool communication, standardising how an agent calls external tools and APIs.  A2A focuses on agent‑to‑agent communication, allowing agents to coordinate and delegate tasks across vendors and frameworks ￼.  In practice, an agent may use MCP to call its calculator tool (e.g., our shared arithmetic server) and A2A to collaborate with other agents on broader tasks ￼.

Steps Automated by A2A

A2A provides built‑in mechanisms for agent discovery, task formatting, secure transport, result streaming and task state management ￼.  It defines the structure of messages and artifacts, uses standard protocols such as HTTP, JSON‑RPC and SSE, supports asynchronous updates, and ensures secure, opaque interaction.  These features remove the need to build custom request/response systems between agents.

Responsibilities for Our Implementation

To leverage A2A in our arithmetic experiment, we must handle several responsibilities:
	•	Publish Agent Cards: Host a /.well-known/agent.json file for each agent describing its capabilities and endpoints ￼.
	•	Implement A2A endpoints: Support JSON‑RPC endpoints like task/send (and SSE endpoints if streaming is needed) so agents can send and receive tasks according to the specification.
	•	Define tasks: Map arithmetic expressions into A2A Task objects with messages (describing the calculation request) and artifacts (carrying the result).
	•	Plan and coordinate: The planner agent must still parse expressions, decide which worker agent to call (using Agent Cards), and sequence tasks appropriately.  Worker agents must call the MCP server to perform their operation and return the result as an artifact.  A2A does not perform planning or computation; it standardises the communication channel.
	•	Handle state transitions: Monitor task states and send additional inputs when required or finalise tasks when complete.  Collect logs and metrics to analyse message counts, latencies, errors, and overall system behaviour.

By integrating these elements, our system aligns with the A2A specification while retaining control over the logic of expression evaluation and performance measurement.

Implementation Overview
----------------------
- `agent_system/Agent.py` defines the shared `IAgent` interface, A2A message models, and an `ArithmeticAgent` capable of executing a single operation by calling the MCP calculator.
- `agent_system/MCP.py` simulates a shared MCP calculator with deterministic `Decimal` arithmetic and exposes helpers to look up operations.
- `agent_system/worker_app.py` hosts a FastAPI service that publishes the agent card at `/.well-known/agent.json` and serves the `/act` endpoint used for A2A `task/send` requests.
- `agent_system/orchestrator.py` discovers worker agents via their agent cards, plans expression evaluation using Python's AST, and delegates each arithmetic step to the appropriate worker via HTTP.

Running The Workers
-------------------
1. Install dependencies with `poetry install`.
2. Start one FastAPI worker per arithmetic operation. Each process reads the `OPERATION_TYPE` environment variable at startup:

   ```bash
   OPERATION_TYPE=add uvicorn agent_system.worker_app:app --host 0.0.0.0 --port 8000
   OPERATION_TYPE=sub uvicorn agent_system.worker_app:app --host 0.0.0.0 --port 8001
   OPERATION_TYPE=mul uvicorn agent_system.worker_app:app --host 0.0.0.0 --port 8002
   OPERATION_TYPE=div uvicorn agent_system.worker_app:app --host 0.0.0.0 --port 8003
   ```

   Each worker advertises its supported intents in `/.well-known/agent.json`, which the orchestrator reads during discovery.

Running The Orchestrator
------------------------
- Launch the orchestrator with the URLs of the running workers. It will fetch each agent card, map intents to `/act`, and then plan the evaluation of the requested expression:

  ```bash
  python -m agent_system.orchestrator "2 + 3 * 4" \
    --agent http://localhost:8000 \
    --agent http://localhost:8001 \
    --agent http://localhost:8002 \
    --agent http://localhost:8003
  ```

  The orchestrator logs its plan internally (see `ArithmeticOrchestrator.execution_log`) so you can inspect which remote agent handled each intermediate result.

Containerisation
----------------
- Use the same image for every worker and select the operation via `OPERATION_TYPE`. A minimal `Dockerfile` might look like this:

  ```dockerfile
  FROM python:3.13-slim
  WORKDIR /app
  COPY pyproject.toml poetry.lock ./
  RUN pip install       "poetry"       && poetry config virtualenvs.create false       && poetry install --no-root --only main
  COPY agent_system ./agent_system
  COPY README.md ./README.md
  ENV OPERATION_TYPE=add
  EXPOSE 8000
  CMD ["uvicorn", "agent_system.worker_app:app", "--host", "0.0.0.0", "--port", "8000"]
  ```

- Build the image once (`docker build -t arithmetic-agent .`) and run a container for each operation, overriding `OPERATION_TYPE` and the port as needed:

  ```bash
  docker run -e OPERATION_TYPE=add -p 8000:8000 arithmetic-agent
  docker run -e OPERATION_TYPE=sub -p 8001:8000 arithmetic-agent
  docker run -e OPERATION_TYPE=mul -p 8002:8000 arithmetic-agent
  docker run -e OPERATION_TYPE=div -p 8003:8000 arithmetic-agent
  ```

- Run the orchestrator in a separate container or on the host. It only needs HTTP access to the worker containers.
