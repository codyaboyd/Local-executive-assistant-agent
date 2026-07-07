FASTCRW_COMPOSE_FILE := docker-compose.fastcrw.yml
FASTCRW_PORT ?= 3002
COMPOSE_ENV_FILE := $(if $(wildcard .env),--env-file .env,)

.PHONY: fastcrw-up fastcrw-down fastcrw-logs fastcrw-health

fastcrw-up:
	FASTCRW_PORT=$(FASTCRW_PORT) docker compose -f $(FASTCRW_COMPOSE_FILE) $(COMPOSE_ENV_FILE) up -d fastcrw

fastcrw-down:
	FASTCRW_PORT=$(FASTCRW_PORT) docker compose -f $(FASTCRW_COMPOSE_FILE) $(COMPOSE_ENV_FILE) down

fastcrw-logs:
	FASTCRW_PORT=$(FASTCRW_PORT) docker compose -f $(FASTCRW_COMPOSE_FILE) $(COMPOSE_ENV_FILE) logs -f fastcrw

fastcrw-health:
	curl -fsS http://127.0.0.1:$(FASTCRW_PORT)/health
