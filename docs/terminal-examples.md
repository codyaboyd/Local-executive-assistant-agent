# Terminal Examples

These examples are text captures that double as smoke-test recipes on a Linux terminal.

## Install and inspect configuration

```console
$ python scripts/dev_make.py install
uv sync --extra dev

$ python scripts/dev_make.py run ARGS="config"
uv run exec-agent config
Executive Assistant Configuration
...
runtime_profile  private-offline
```

## CPU-safe local mode

```console
$ cp .env.cpu.example .env
$ python scripts/dev_make.py run ARGS="profile list"
uv run exec-agent profile list
Runtime Profiles
*  cpu-safe  sshleifer/tiny-gpt2  cpu  False
```

## GPU mode when CUDA is available

```console
$ nvidia-smi
# NVIDIA driver table appears here
$ cp .env.gpu.example .env
$ uv run python -c "import torch; print(torch.cuda.is_available())"
True
$ python scripts/dev_make.py run ARGS="model-test 'Draft a two-line daily brief.'"
uv run exec-agent model-test 'Draft a two-line daily brief.'
...
```

## Offline eval harness

```console
$ python scripts/dev_make.py eval
uv run exec-agent eval run
Executive Assistant Eval Results
PASS  memory_retrieval_budget_context
PASS  pdf_qa_receipt_threshold
PASS  docx_qa_approval_owner
```

## Sample document workflow

```console
$ uv run exec-agent task summarize-notes --file docs/samples/board-prep-notes.md
Notes Summary
...

$ uv run exec-agent task action-items --file docs/samples/travel-policy.txt
Action Items
...
```
