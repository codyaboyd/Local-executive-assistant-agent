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

Run the placeholder chat command:

```bash
uv run python -m exec_agent chat
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
