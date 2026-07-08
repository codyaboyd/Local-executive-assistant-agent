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

## Local Linux Installation

The package exposes a console script named `exec-agent` from `pyproject.toml`, so a local install makes the assistant available without typing `python -m exec_agent`.

1. Install Linux prerequisites. On Debian or Ubuntu:

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip git build-essential
```

2. Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec "$SHELL"
uv --version
```

3. Clone the project and enter the repository:

```bash
git clone <repository-url>
cd Local-executive-assistant-agent
```

4. Create local configuration:

```bash
cp .env.example .env
```

5. Install the project in editable mode with development tools:

```bash
uv sync --extra dev
uv pip install -e .
```

6. Verify that the installed command is on your PATH:

```bash
exec-agent --help
```

If your shell cannot find `exec-agent`, run it through uv from the repository (`uv run exec-agent --help`) or ensure the active virtual environment's `bin` directory is on your PATH.

### Installed Command Examples

After installation, these commands are available directly from your shell:

```bash
exec-agent chat
exec-agent ingest pdf ./file.pdf
exec-agent ask "question"
exec-agent web search "query"
```

Additional useful commands:

```bash
exec-agent ingest docx ./file.docx
exec-agent rag search "topic"
exec-agent sessions list
exec-agent config
```

## GPU Setup Notes

GPU support is optional. The assistant can run on CPU, but local text, embedding, and vision models are much faster with a supported NVIDIA GPU.

- Install a recent NVIDIA driver on Linux and confirm the GPU is visible:

```bash
nvidia-smi
```

- Install or sync dependencies after the driver is available so PyTorch can detect CUDA correctly:

```bash
uv sync --extra dev
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

- Use the GPU-oriented runtime profile when CUDA is available:

```bash
exec-agent profile use gpu-fast
exec-agent config
```

- Image commands accept an explicit device flag:

```bash
exec-agent image describe ./image.png --device cuda
exec-agent image ask ./image.png "what is in this image?" --device cuda
```

- If CUDA is not available, use the CPU-safe profile and `--device cpu` for image commands:

```bash
exec-agent profile use cpu-safe
exec-agent image describe ./image.png --device cpu
```

## Troubleshooting

### `exec-agent: command not found`

Install the package in the active environment or run it through uv:

```bash
uv pip install -e .
uv run exec-agent --help
```

If you installed into a virtual environment, activate it before running `exec-agent`.

### Import errors after installation

Refresh the environment and reinstall the editable package:

```bash
uv sync --extra dev
uv pip install -e .
```

Then verify imports with:

```bash
uv run python -c "import exec_agent; print(exec_agent.__version__)"
```

### PDF or DOCX ingestion fails

Check that the file exists, is readable, and has a supported extension. Then rerun with an explicit path:

```bash
exec-agent ingest pdf ./file.pdf
exec-agent ingest docx ./file.docx
```

### No document context found for `ask`

Ingest documents before asking questions, and make sure you are using the same `EXEC_AGENT_DATA_DIR` and `EXEC_AGENT_VECTOR_DB_PATH` as the ingest step:

```bash
exec-agent ingest pdf ./file.pdf
exec-agent ask "question"
exec-agent config
```

### Web search cannot connect to FastCRW

Start the local FastCRW service and check health:

```bash
make fastcrw-up
exec-agent web health
exec-agent web search "query"
```

Confirm `FASTCRW_BASE_URL`, `FASTCRW_PORT`, and `FASTCRW_API_PREFIX` in `.env` if the health check still fails.

### CUDA/GPU is not detected

Confirm the NVIDIA driver works, then check PyTorch CUDA detection:

```bash
nvidia-smi
uv run python -c "import torch; print(torch.cuda.is_available())"
```

If CUDA is unavailable, switch to CPU mode:

```bash
exec-agent profile use cpu-safe
```

### Local model downloads are slow or fail

The first run may download Hugging Face models. Confirm network access and available disk space, or configure a local Hugging Face cache before retrying.

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
