# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EnterpriseOps-Gym is a benchmark for evaluating LLM agents on stateful, multi-step enterprise workflows. It provides 1,150 tasks across 8 domains (Calendar, CSM, Drive, Email, HR, ITSM, Teams, Hybrid), with 512 tools backed by 164 database tables. Agents interact with gym servers via the MCP protocol; results are validated by a verifier engine that checks database state, LLM-judged responses, and tool execution.

## Commands

### Setup
```bash
uv sync --extra all          # install all provider dependencies
cp -r conf.example/ conf/    # create config directory
# Edit conf/llm/<model>.json with API key and model details
```

### Run Tests
```bash
.venv/bin/pytest tests/                          # all tests
.venv/bin/pytest tests/ -m "not integration"     # skip live API tests
.venv/bin/pytest tests/test_llm_client_divyam.py # single file
```

### Run Benchmark
```bash
# Single domain, direct execution
python evaluate.py \
    --hf_dataset ServiceNow-AI/EnterpriseOps-Gym \
    --domain teams --mode oracle \
    --llm_config conf/llm/my-model.json \
    --output_folder results/react/my-model/teams/oracle \
    --orchestrator react \
    --concurrency 4 --num_runs 1

# Parallel execution via Ray
python ray_experiment_queue.py --experiment_config conf/ray/experiment.json

# Score results
python compute_score.py --results_folder results/react/my-model/teams/oracle
```

## Architecture

### Execution Flow

```
evaluate.py / ray_experiment_queue.py
    → BenchmarkExecutor (benchmark/executor.py)
        → MCPClient (benchmark/mcp_client.py)     — creates/seeds DB, routes tool calls
        → LLMClient (benchmark/llm_client.py)      — multi-provider LLM interface
        → AgentOrchestrator (orchestrators/)        — controls agent loop
        → Verifier (benchmark/verifier.py)          — validates final state
```

### Key Abstractions

**`benchmark/models.py`** — All shared data structures: `BenchmarkConfig`, `LLMConfig`, `GymServerConfig`, `VerifierConfig`, `MCPToolCall`, `MCPToolResponse`.

**`benchmark/llm_client.py`** — Unified LLM client wrapping LangChain. Supports Anthropic, OpenAI/Azure, Google/Vertex, Bedrock, DeepSeek, Divyam, vLLM, OpenRouter, Qwen. Provider is selected from `LLMConfig.provider`. Supports reasoning models via `effort`/`reasoning` fields.

**`benchmark/mcp_client.py`** — HTTP JSON-RPC client for gym MCP servers. Handles DB lifecycle (create/seed/delete) and routes tool calls to the correct gym when multiple gyms are active.

**`benchmark/executor.py`** — `BenchmarkExecutor` owns the full loop: load tools from all gyms → merge → call LLM → dispatch tool calls → collect observations → repeat → verify.

**`benchmark/verifier.py`** — Three verifier types (`VerifierType` enum):
- `database_state`: runs SQL against the final DB and compares with expected value
- `response_check`: uses an LLM judge to evaluate the agent's final response
- `tool_execution`: checks whether specific tools were called

### Orchestrators (`orchestrators/`)

All inherit from `AgentOrchestrator` (`orchestrators/base.py`).

| Orchestrator | Class | Strategy |
|---|---|---|
| `react` | `ReactOrchestrator` | Standard ReAct loop |
| `planner_react` | `PlannerReactOrchestrator` | Planner LLM generates plan → executor LLM runs ReAct with plan as context |
| `decomposing_planner` | `DecomposingPlannerOrchestrator` | Decompose into 2–5 subtasks → independent ReAct per subtask with shared working memory → aggregate |

### Configuration (`conf/`)

- `conf/llm/<model>.json` — Provider, model ID, API key, endpoint, temperature, max tokens, reasoning settings
- `conf/ray/experiment.json` — Which domains, modes, LLM configs, and orchestrator to run
- `conf/ray/domain_conf.json` — Domain → MCP server endpoint mapping
- `conf/ray/llm_concurrency.json` — Per-model concurrency limits
- `conf/ray/base_env.json` — Base environment variables for Ray workers

### Adding a New LLM Provider

1. Add an entry to `LLMConfig` provider handling in `benchmark/llm_client.py`
2. Add the provider's package as an optional dependency in `pyproject.toml`
3. Create a config file in `conf/llm/`
4. Add tests in `tests/` (mark live API calls with `@pytest.mark.integration`)

The Divyam provider (`conf/llm/divyam.json`, `tests/test_llm_client_divyam.py`) is the reference example for a custom provider. Note: the scoring logic skips HTTP 429 responses from this provider.
