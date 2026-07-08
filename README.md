# Local Executive Assistant AI Agent

A local-first Linux terminal AI executive assistant for private executive workflows. It runs from the command line, stores memory and document context locally, supports CPU and CUDA runtime profiles, and includes deterministic evals so you can verify the assistant without network access or heavyweight model calls.

## What it can do

- Interactive terminal chat with persistent named sessions.
- Executive workflows for note summaries, email drafts, meeting briefs, action items, topic research, and daily briefings.
- Local long-term memory backed by SQLite.
- Local RAG over PDF and DOCX documents backed by ChromaDB and sentence-transformers.
- Local image description and visual Q&A for PNG, JPG, JPEG, and WEBP files.
- Optional self-hosted FastCRW web search/scrape/crawl integration.
- Runtime profiles for CPU-safe, GPU-fast, private offline, research online, and HITL test modes.
- Offline eval harness with mocked tools.

## Quick start

```bash
git clone <repository-url>
cd Local-executive-assistant-agent
python scripts/dev_make.py install
cp .env.cpu.example .env
python scripts/dev_make.py run ARGS="config"
python scripts/dev_make.py eval
```

Start the chat UI:

```bash
python scripts/dev_make.py run
# or
uv run exec-agent chat
```

## Requirements

- Linux terminal environment.
- Python 3.11 or newer.
- [`uv`](https://docs.astral.sh/uv/) for dependency management.
- Optional NVIDIA GPU with working CUDA driver for GPU mode.
- Optional Docker Compose for self-hosted FastCRW.

## Generated Makefile commands

The repository keeps product workflow targets out of the checked-in Makefile. Use `scripts/dev_make.py` to generate a temporary Makefile on the fly and delegate to `make`.

| Command | Purpose |
| --- | --- |
| `python scripts/dev_make.py install` | Sync runtime and development dependencies with `uv sync --extra dev`. |
| `python scripts/dev_make.py test` | Run the pytest suite. |
| `python scripts/dev_make.py run` | Generates a temporary Makefile and runs `exec-agent`; defaults to `chat`. Pass subcommands with `ARGS="..."`. |
| `python scripts/dev_make.py eval` | Run offline deterministic assistant evals. |
| `python scripts/dev_make.py fastcrw-up` | Start the optional self-hosted FastCRW container. |
| `python scripts/dev_make.py fastcrw-down` | Stop FastCRW. |
| `python scripts/dev_make.py fastcrw-health` | Check FastCRW health. |

Examples:

```bash
python scripts/dev_make.py run ARGS="profile list"
python scripts/dev_make.py run ARGS="task summarize-notes --file docs/samples/board-prep-notes.md"
```

## Environment files

Copy the profile that matches your machine and workflow:

```bash
cp .env.cpu.example .env       # normal Linux CPU-only mode
cp .env.gpu.example .env       # CUDA mode when torch can see CUDA
cp .env.research.example .env  # online research via self-hosted FastCRW
```

The original `.env.example` remains a compact reference for all common variables.

## CPU mode

CPU mode is the safest default for a normal Linux machine:

```bash
cp .env.cpu.example .env
python scripts/dev_make.py run ARGS="profile list"
python scripts/dev_make.py run ARGS="model-test 'Write a one sentence briefing.'"
```

The CPU profile uses `EXEC_AGENT_RUNTIME_PROFILE=cpu-safe`, `EXEC_AGENT_DEVICE=cpu`, disables web access, and writes local data under `~/.local/share/exec-agent` unless overridden.

## GPU mode

GPU mode is optional and only requires CUDA when you choose it:

```bash
nvidia-smi
cp .env.gpu.example .env
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
python scripts/dev_make.py run ARGS="model-test 'Draft a short executive update.'"
```

`EXEC_AGENT_DEVICE=auto` selects CUDA when available and CPU otherwise. If `cuda` is requested but CUDA is unavailable, the model adapter warns and falls back to CPU so commands fail safely instead of crashing.

## CLI command examples

```bash
uv run exec-agent config
uv run exec-agent profile list
uv run exec-agent profile use cpu-safe
uv run exec-agent chat --session work
uv run exec-agent sessions list
uv run exec-agent memory add "User prefers concise board updates" --tag preference
uv run exec-agent memory search board
uv run exec-agent task action-items --file docs/samples/board-prep-notes.md
uv run exec-agent task draft-email "ask finance for budget deltas" --tone warm
uv run exec-agent task action-items --file docs/samples/travel-policy.txt
uv run exec-agent eval run
```

For full terminal captures, see [`docs/terminal-examples.md`](docs/terminal-examples.md).

## Sample documents

Test fixtures live in [`docs/samples/`](docs/samples/):

- `board-prep-notes.md` for task workflows.
- `travel-policy.txt` for simple text prompts.

## Optional web research

Online research is intentionally opt-in and expects a self-hosted FastCRW server:

```bash
cp .env.research.example .env
python scripts/dev_make.py fastcrw-up
python scripts/dev_make.py fastcrw-health
uv run exec-agent web search "AI market update" --max-results 3
```

Private and CPU-safe profiles disable web access. `EXEC_AGENT_LOCAL_ONLY=true` always disables web, scraping, and crawling regardless of profile.

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the final architecture, runtime layers, data flow, CPU/GPU behavior, safety model, and extension points.

## Development checks

```bash
python scripts/dev_make.py test
python scripts/dev_make.py eval
uv run exec-agent --help
uv run exec-agent config
```
