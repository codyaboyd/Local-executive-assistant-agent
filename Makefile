FASTCRW_COMPOSE_FILE := docker-compose.fastcrw.yml
FASTCRW_PORT ?= 3002
COMPOSE_ENV_FILE := $(if $(wildcard .env),--env-file .env,)
ARGS ?= chat
WEB_ARGS ?= serve

.PHONY: install test run web fastcrw-up fastcrw-down fastcrw-logs fastcrw-health eval

install:
	uv sync --extra dev

test:
	uv run pytest

run:
	uv run exec-agent $(ARGS)

web:
	uv run exec-agent web $(WEB_ARGS)

fastcrw-up:
	FASTCRW_PORT=$(FASTCRW_PORT) docker compose -f $(FASTCRW_COMPOSE_FILE) $(COMPOSE_ENV_FILE) up -d fastcrw

fastcrw-down:
	FASTCRW_PORT=$(FASTCRW_PORT) docker compose -f $(FASTCRW_COMPOSE_FILE) $(COMPOSE_ENV_FILE) down

fastcrw-logs:
	FASTCRW_PORT=$(FASTCRW_PORT) docker compose -f $(FASTCRW_COMPOSE_FILE) $(COMPOSE_ENV_FILE) logs -f fastcrw

fastcrw-health:
	curl -fsS http://127.0.0.1:$(FASTCRW_PORT)/health

eval:
	uv run exec-agent eval run
