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
uv run exec-agent task run "prepare a board meeting checklist"
uv run exec-agent task run "prepare a board meeting checklist" --autonomous
uv run exec-agent task status
uv run exec-agent task cancel <task_id>
uv run exec-agent task history
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

## Autonomous task execution

The autonomous task runner persists plans, step outputs, errors, and final summaries to SQLite so terminal commands and web UI clients can stream or poll the same task progress source. Configure autonomy with:

```bash
EXEC_AGENT_AUTONOMY_LEVEL=human_approved
EXEC_AGENT_MAX_AUTONOMOUS_STEPS=25
EXEC_AGENT_REQUIRE_APPROVAL_FOR_DANGEROUS_COMMANDS=true
EXEC_AGENT_TASK_TIMEOUT_SECONDS=1800
```

Supported autonomy levels are `off`, `suggest_only`, `human_approved`, `autonomous_limited`, and `autonomous_full`. The default is `human_approved`; use `exec-agent task run "..." --autonomous` to opt into full autonomous execution for a single run. Loop safeguards stop tasks when they repeat results, exceed the configured step budget, hit the timeout, or select a dangerous tool while approvals are required.

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the final architecture, runtime layers, data flow, CPU/GPU behavior, safety model, and extension points.

## Development checks

```bash
python scripts/dev_make.py test
python scripts/dev_make.py eval
uv run exec-agent --help
uv run exec-agent config
```

## Small-GPU model registry and role switching

The assistant includes a curated open-source model registry for consumer GPUs with 16GB VRAM or less, plus CPU-only mode. Configure the selector with:

```bash
EXEC_AGENT_MODEL_PRESET=default        # default|low_vram|cpu_only|quality|coding|research
EXEC_AGENT_MODEL_AUTO_PULL=false       # true pulls selected defaults during setup workflows
EXEC_AGENT_MAX_VRAM_GB=16
EXEC_AGENT_GENERAL_MODEL_ID=auto
EXEC_AGENT_CODING_MODEL_ID=auto
EXEC_AGENT_SUMMARY_MODEL_ID=auto
EXEC_AGENT_DOCQA_MODEL_ID=auto
EXEC_AGENT_RESEARCH_MODEL_ID=auto
EXEC_AGENT_TOOL_MODEL_ID=auto
EXEC_AGENT_EMBEDDING_MODEL_ID=auto
EXEC_AGENT_VISION_MODEL_ID=auto
```

Use `auto` or `default` to let the registry choose a role-specific model. Set any role variable to a Hugging Face model ID to override it. Runtime workflows pass a task role to the model adapter, so summarization, document Q&A, web research, coding, tool-calling, embeddings, and vision can use specialist defaults instead of one global model.

Model management commands:

```bash
uv run exec-agent models list
uv run exec-agent models status
uv run exec-agent models pull-defaults
uv run exec-agent models pull --role coding
uv run exec-agent models set-role coding Qwen/Qwen2.5-Coder-3B-Instruct
uv run exec-agent models benchmark
```

`pull-defaults` only pulls the registry selections for the active preset and de-duplicates shared models. The CLI warns when a selected model recommends more VRAM than `EXEC_AGENT_MAX_VRAM_GB`; avoid pulling models above your hardware budget unless you intentionally plan to run them on CPU or another backend. The loader degrades gracefully: if the selected GPU model cannot load, it retries CPU for that model, then smaller role-compatible models, then CPU-friendly fallbacks.

### Recommended models by GPU size

| Hardware budget | Preset | General reasoning | Coding | Embeddings | Vision | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| CPU / low VRAM | `cpu_only` | Qwen/Qwen2.5-1.5B-Instruct or TinyLlama/TinyLlama-1.1B-Chat-v1.0 fallback | Qwen/Qwen2.5-Coder-1.5B-Instruct | sentence-transformers/all-MiniLM-L6-v2 | Salesforce/blip-image-captioning-base | Prefer CPU-friendly models, small context windows, and conservative max token settings. |
| 8GB VRAM | `default` or `low_vram` | Qwen/Qwen2.5-3B-Instruct, with Phi-3.5 Mini and Gemma 2 2B available as lightweight alternatives | Qwen/Qwen2.5-Coder-3B-Instruct | BAAI/bge-small-en-v1.5 | Qwen/Qwen2-VL-2B-Instruct when it fits; BLIP fallback otherwise | Best small-GPU balance for local executive workflows. |
| 12GB VRAM | `default` | Qwen/Qwen2.5-7B-Instruct in a 4-bit/quantized runtime | Qwen/Qwen2.5-Coder-3B-Instruct | BAAI/bge-small-en-v1.5 | Qwen/Qwen2-VL-2B-Instruct | Use 7B reasoning models when quantized; keep coding on 3B unless using the 16GB/coding preset. |
| 16GB VRAM | `quality` or `coding` | Qwen/Qwen2.5-7B-Instruct or Mistral-7B-Instruct quantized | Qwen/Qwen2.5-Coder-7B-Instruct quantized | BAAI/bge-base-en-v1.5 for quality, BAAI/bge-small-en-v1.5 by default | Qwen/Qwen2-VL-2B-Instruct | Strongest local tier in the curated registry while staying inside the consumer-GPU target. |

Summarization and document QA use the active general reasoning tier by default, and `EXEC_AGENT_SUMMARY_MODEL_ID` can point to a specialized summarization model when a workflow needs one. Tool calling prefers the strongest instruction-following Qwen model that fits the active preset and VRAM budget.

## Bootstrap web UI

The project includes a FastAPI-backed, Bootstrap 5 remote web interface for operating the assistant from a browser without React.

### Start the web UI

```bash
uv run exec-agent-web
# or
uv run uvicorn exec_agent.web:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>. To require login, set a local web password before starting the server:

```bash
export EXEC_AGENT_WEB_PASSWORD='choose-a-local-password'
uv run exec-agent-web
```

### Routes and capabilities

| Route | Capability |
| --- | --- |
| `/login`, `/logout` | Optional password-gated browser session. |
| `/` | Mobile-friendly dashboard with dark mode and quick links. |
| `/chat` | Streaming assistant chat over Server-Sent Events. |
| `/tasks`, `/tasks/{task_id}` | Run autonomous tasks, view progress, and approve/reject HITL action cards. |
| `/files` | Browse configured allowed directories and upload PDF, DOCX, image, Markdown, or text files. |
| `/ingest` | Ingest uploaded or local PDF/DOCX/text/image files into the vector database. |
| `/memory` | Search and manage long-term SQLite memories. |
| `/models` | View model registry entries and active model selections/status. |
| `/settings` | Inspect runtime settings and choose the active runtime profile for the next restart. |
| `/web` | Run self-hosted FastCRW web searches and view service health. |
| `/shell` | Run allowlisted shell commands inside the configured shell workspace and view command history. |

The UI uses Bootstrap responsive components, a persistent dark-mode toggle, and Rich-inspired readable cards/preformatted output for chat, task, and shell streams. File browsing, shell execution, FastCRW, uploads, and ingestion continue to respect the same safety settings used by the CLI, including allowed directories, upload extensions, command allowlists, and autonomy/HITL gates.
