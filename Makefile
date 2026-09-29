.DEFAULT_GOAL := help
COMPOSE := docker compose

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

up: ## Build and start the stack
	$(COMPOSE) up -d --build

model: ## Pull the generation model into Ollama
	$(COMPOSE) exec ollama ollama pull $${LLM_MODEL:-qwen3:8b}

ingest: ## Index everything in CORPUS_DIR
	$(COMPOSE) run --rm --entrypoint python ingest ingest.py /data

reindex: ## Re-parse and re-index everything
	$(COMPOSE) run --rm --entrypoint python ingest ingest.py /data --reindex

health: ## Show component status
	@curl -s localhost:8080/health | python3 -m json.tool

logs: ## Tail all logs
	$(COMPOSE) logs -f

vram: ## What is currently on the GPU
	nvidia-smi --query-gpu=memory.used,memory.total --format=csv

down: ## Stop the stack (data is preserved)
	$(COMPOSE) down

nuke: ## Stop and delete all indexed data and model caches
	$(COMPOSE) down -v

.PHONY: help up model ingest reindex health logs vram down nuke
