# Architecture

## Goals

Local Executive Assistant Agent is a Linux terminal assistant optimized for private, local-first executive workflows. It provides chat, long-term memory, document retrieval, image analysis, optional self-hosted web research, and an offline eval harness without requiring hosted AI services.

## Runtime layers

1. **CLI boundary (`src/exec_agent/cli.py`)** exposes Typer commands for chat, config, profiles, memory, RAG, ingestion, images, web research, task workflows, sessions, and evals.
2. **Configuration (`src/exec_agent/config.py`)** loads `.env` and environment variables through Pydantic Settings. Runtime profiles select safe CPU defaults, CUDA defaults, private offline behavior, online research behavior, and HITL test behavior.
3. **Chat orchestration (`src/exec_agent/chat.py`, `app/graph/*`)** routes a user turn through graph nodes, optional approval gates, memory context, RAG context, and streaming model output.
4. **Local model adapters (`src/exec_agent/models/llm.py`, `app/tools/image.py`)** use Hugging Face Transformers. Device resolution supports `cpu`, `cuda`, and `auto`; CUDA requests fall back safely to CPU when CUDA is unavailable.
5. **Knowledge stores (`app/memory/*`)** persist long-term memory in SQLite and RAG chunks in ChromaDB under the configured data directory.
6. **Tool adapters (`app/tools/*`)** ingest PDFs and DOCX files, analyze images, and call a user-configured self-hosted FastCRW instance for web search/scrape/crawl.
7. **Evaluation (`app/evals/runner.py`)** provides deterministic mocked evals that are safe for CI, CPU-only machines, and offline development.

## Data flow

```text
Terminal command
  -> Typer CLI
  -> Settings/runtime profile
  -> Task/chat/document/image/web handler
  -> Local model and/or local stores and/or self-hosted FastCRW
  -> Rich terminal output
```

## CPU and GPU modes

- CPU mode uses `EXEC_AGENT_RUNTIME_PROFILE=cpu-safe` and `EXEC_AGENT_DEVICE=cpu`. It is intended to work on normal Linux machines without NVIDIA drivers.
- GPU mode uses `EXEC_AGENT_RUNTIME_PROFILE=gpu-fast` and `EXEC_AGENT_DEVICE=cuda`. It accelerates Transformers pipelines when PyTorch reports CUDA availability.
- `EXEC_AGENT_DEVICE=auto` selects CUDA when available and CPU otherwise.
- If CUDA is explicitly requested but unavailable, the model adapter warns and falls back to CPU instead of crashing.

## Safety model

- Private profiles disable web and FastCRW.
- `EXEC_AGENT_LOCAL_ONLY=true` forces web, scraping, and crawling off regardless of profile.
- Crawls can require human approval with `FASTCRW_CRAWL_REQUIRES_APPROVAL=true` or HITL profiles.
- Executive task workflows draft text only; they do not send email, change calendars, or modify external systems.

## Extension points

- Add new CLI workflows under `task_app` in `src/exec_agent/cli.py`.
- Add new graph behavior in `app/graph/nodes.py` and wire it through `app/graph/builder.py`.
- Add tool integrations under `app/tools/` and keep side effects behind explicit CLI commands or HITL gates.
- Add deterministic eval coverage in `app/evals/runner.py` before introducing heavyweight or online behavior.
