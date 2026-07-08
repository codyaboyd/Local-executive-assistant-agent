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

| GPU VRAM | Preset | Recommended roles/models | Notes |
| --- | --- | --- | --- |
| 4GB | `low_vram` or `cpu_only` | Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-Coder-1.5B-Instruct, sentence-transformers/all-MiniLM-L6-v2 | Prefer CPU-friendly or quantized runtimes; keep context and max tokens small. |
| 6GB | `low_vram` | Qwen/Qwen2.5-1.5B-Instruct for general/research/doc QA; Qwen/Qwen2.5-Coder-1.5B-Instruct for coding | Good for light executive workflows without downloading huge models. |
| 8GB | `default` | Qwen/Qwen2.5-3B-Instruct, Qwen/Qwen2.5-Coder-3B-Instruct, microsoft/Phi-3.5-mini-instruct | Best balance for instruction following, grounded reasoning, and specialist switching. |
| 12GB | `default` or `research` | Qwen/Qwen2.5-3B-Instruct, Phi-3.5-mini-instruct, Hermes-3-Llama-3.2-3B | Use research preset when web synthesis and tool-style prompts matter most. |
| 16GB | `quality` | Qwen/Qwen2.5-7B-Instruct for general reasoning, Qwen/Qwen2.5-Coder-3B-Instruct for coding | Higher quality while staying inside the consumer-GPU target; monitor VRAM before increasing context. |

CPU-only machines should use `EXEC_AGENT_MODEL_PRESET=cpu_only` and `EXEC_AGENT_DEVICE=cpu`. Embeddings and vision defaults remain small and CPU-capable, but image tasks may be slower without CUDA.
