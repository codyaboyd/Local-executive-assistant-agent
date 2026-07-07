# Local Executive Assistant AI Agent

A Python 3.11+ Linux terminal-based scaffold for a local-first AI executive assistant. This initial version provides the project structure, CLI shell, configuration loading, terminal UI components, and tests needed to begin building the full assistant.

> This is an initial scaffold only. Full agent capabilities such as calendar integration, email drafting, document retrieval, and model orchestration are intentionally not implemented yet.

## Features in This Scaffold

- Python 3.11+ package using a `src/` layout
- Dependency management with `uv` via `pyproject.toml`
- Typer-powered CLI commands
- Rich terminal output
- Pydantic Settings configuration loading from environment variables and `.env`
- Example environment file at `.env.example`
- Basic pytest coverage for CLI and configuration behavior
- PDF ingestion into local vector storage with source/page metadata
- DOCX ingestion into local vector storage with source/section metadata
- Question answering over uploaded PDFs and DOCX files with page or section references when available
- Local image description and visual question answering for PNG, JPG, JPEG, and WEBP files, with generated context stored in vector search
- Module entrypoint that runs with:

```bash
python -m exec_agent chat
```

## Repository Structure

```text
.
├── .env.example
├── README.md
├── pyproject.toml
├── src
│   └── exec_agent
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       └── config.py
└── tests
    ├── test_cli.py
    └── test_config.py
```

## Requirements

- Linux terminal environment
- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/) for dependency management

## Setup

1. Install dependencies:

```bash
uv sync --extra dev
```

2. Create a local environment file:

```bash
cp .env.example .env
```

3. Adjust `.env` values as needed:

```dotenv
EXEC_AGENT_ENV=development
EXEC_AGENT_LOG_LEVEL=INFO
EXEC_AGENT_DATA_DIR=~/.local/share/exec-agent
```

## Usage

Run the chat command:

```bash
uv run python -m exec_agent chat
```

Start or resume a persistent SQLite-backed chat session:

```bash
uv run python -m exec_agent chat --session work
```

Manage saved chat sessions:

```bash
uv run python -m exec_agent sessions list
uv run python -m exec_agent sessions show work
uv run python -m exec_agent sessions delete work
```

Ingest a PDF into local RAG storage:

```bash
uv run python -m exec_agent ingest pdf ./file.pdf
```

Ingest a DOCX into local RAG storage:

```bash
uv run python -m exec_agent ingest docx ./file.docx
```

Ask a question about ingested PDFs and DOCX files:

```bash
uv run python -m exec_agent ask "question about uploaded documents"
```

Describe an image and save the description as searchable vector context:

```bash
uv run python -m exec_agent image describe ./image.png
```

Ask a question about an image and save the answer as searchable vector context:

```bash
uv run python -m exec_agent image ask ./image.png "what is shown here?"
```

Image commands use local Hugging Face vision-language models. Use `--device cpu`, `--device cuda`, or `--device auto` to control CPU/GPU inference, and `--model` to select a different local-compatible model.

You can also inspect the effective configuration:

```bash
uv run python -m exec_agent config
```

If the package is installed, the console script is available as:

```bash
uv run exec-agent chat
```

## Testing

Run the test suite with:

```bash
uv run pytest
```

## Development Notes

The scaffold is intentionally small and focused. Future work can add agent orchestration, model providers, local memory, calendar/email integrations, retrieval, and approval workflows while preserving the CLI-first foundation.

## License

Add your preferred license for this project.

## Using Self-Hosted FastCRW

This project can use a self-hosted FastCRW server for web research. The default
configuration targets a local server and does **not** use hosted FastCRW by
default. The assistant app does not require FastCRW to be running at startup; if
FastCRW is offline, only web commands/features fail gracefully with a clear
FastCRW connection error.

### Docker Compose setup

The repository includes `docker-compose.fastcrw.yml` for running the official
self-hostable FastCRW image (`ghcr.io/us/crw:latest`) on localhost. The container
listens internally on port 3000 and is published to `127.0.0.1:${FASTCRW_PORT}`
(default `3002`).

```bash
cp .env.example .env
make fastcrw-up
exec-agent web health
```

Useful Make targets:

```bash
make fastcrw-up      # start FastCRW in the background
make fastcrw-down    # stop and remove the FastCRW container
make fastcrw-logs    # follow FastCRW logs
make fastcrw-health  # call http://127.0.0.1:${FASTCRW_PORT}/health
```

Set `FASTCRW_PORT=3002` in `.env` or when invoking `make` to choose the local
port. `FASTCRW_API_KEY` is optional for self-hosted FastCRW and may be left blank.

### Configuration

Select a runtime profile with `EXEC_AGENT_RUNTIME_PROFILE`, or manage profiles from the CLI:

```bash
python -m exec_agent profile list
python -m exec_agent profile use cpu-safe
```

Profiles control the text model id, device mode, web access, human-in-the-loop (HITL) approvals, vector database path, and logging verbosity:

- `cpu-safe`: CPU-only local execution with web access and HITL disabled. Uses a profile-scoped vector DB under the data directory.
- `gpu-fast`: CUDA/GPU local execution for faster inference with web access and HITL disabled. Uses its own profile-scoped vector DB.
- `private-offline`: local-only mode with web access and FastCRW disabled.
- `research-online`: enables web access and self-hosted FastCRW (`web_enabled=true`, `fastcrw_enabled=true`) using `FASTCRW_BASE_URL`.
- `test-hitl`: enables web access, self-hosted FastCRW, and human-in-the-loop approvals; crawl operations require approval before running.

Set `EXEC_AGENT_VECTOR_DB_PATH` to override the vector database path for any profile.

Set these environment variables as needed:

```bash
EXEC_AGENT_RUNTIME_PROFILE=private-offline
FASTCRW_PORT=3002
FASTCRW_BASE_URL=http://localhost:3002
FASTCRW_API_PREFIX=/v1
FASTCRW_API_KEY=              # optional for your self-hosted server
FASTCRW_TIMEOUT_SECONDS=30
FASTCRW_MAX_RESULTS=5
FASTCRW_ENABLE_SCRAPE=true
FASTCRW_ENABLE_CRAWL=false
```

`FASTCRW_API_KEY` is optional and is sent as both a bearer token and `X-API-Key`
header when present. Crawling is disabled by default because it can touch many
pages; enable it only for servers and sites you are allowed to crawl. The
assistant defaults to self-hosted FastCRW only and does not call hosted search
APIs unless you explicitly configure a hosted-compatible endpoint yourself.

### CLI commands

```bash
exec-agent web health
exec-agent web search "quarterly market outlook"
exec-agent web scrape "https://example.com"
exec-agent web crawl "https://example.com" --limit 10
```

Scraped and crawled page content is stored in the local vector database with
metadata including `url`, `title`, `fetched_at`, `source_type=web`, and
`provider=fastcrw_self_hosted`.

### LangGraph behavior and approvals

LangGraph first classifies each user turn as `general_chat`,
`document_question`, `web_research`, `image_question`, `memory_update`, or
`task_planning`, then routes the turn to the matching tool node before approval
and local LLM response generation. If the classifier cannot choose a route with
sufficient confidence, the graph uses a safe general-chat fallback that asks for
clarification when needed. Every tool node prints a `TOOL CALL: ...` line and
records the call in graph state so terminal sessions can observe routing and tool
usage.

LangGraph uses FastCRW only when web access is enabled in graph state and either
the user explicitly asks for web research or the active profile allows online
research. In human-in-the-loop mode, crawl operations require approval and show
the target domain plus the maximum page limit; the operator can approve, reject,
or edit the limit before the crawl runs.

FastCRW errors are surfaced clearly for offline servers, invalid API keys,
timeouts, empty search results, blocked scrapes, and crawl limit violations.
