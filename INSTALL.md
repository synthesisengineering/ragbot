# Installing Ragbot

Ragbot — the open-source reference runtime for conversational synthesis engineering.

This guide gets a working stack up on a single machine. For the configuration that follows — API keys, providers, workspace paths — see [CONFIGURE.md](CONFIGURE.md).

## Prerequisites

- **Docker** with Compose v2 (Docker Desktop on macOS / Windows, or Docker Engine on Linux).
- **Git**, for cloning the repository.
- **At least one API key** for a hosted LLM provider (Anthropic, OpenAI, or Google), or a running [Ollama](https://ollama.com) on the host for local models. The Ollama path needs no API key.

A Python toolchain is **not** required for the default install. If you intend to run the CLI directly on the host instead of inside the container, install **Python 3.14 or newer** from [python.org](https://www.python.org/downloads/) and use the project's `requirements.txt`.

## Install via Docker (default)

```bash
git clone https://github.com/synthesisengineering/ragbot.git
cd ragbot
cp .env.docker .env
# Edit .env and add at least one provider API key.

docker compose up -d
```

The web UI is at <http://localhost:3000>. The API is at <http://localhost:8000>. The bundled Postgres + pgvector service starts automatically.

To stop the stack:

```bash
docker compose down
```

## Try demo mode

Demo mode runs against a bundled workspace and skill, hard-isolated from anything on the host. No keys are required beyond a provider API key for the LLM itself.

```bash
RAGBOT_DEMO=1 docker compose up -d
# Open http://localhost:3000 — a yellow demo-mode banner appears in the UI.
```

Unset `RAGBOT_DEMO` and bring the stack up again to return to your real workspaces.

## Point Ragbot at your data

Ragbot discovers `ai-knowledge-*` workspaces on the host. The default Docker Compose mounts `~/workspaces/` read-only into the container. Override paths in `docker-compose.override.yml`:

```bash
cp docker-compose.override.example.yml docker-compose.override.yml
# Edit to mount your knowledge directories.
```

See [README.md](README.md) for the workspace and `ai-knowledge` layout, and the configuration guide for details on `~/.synthesis/console.yaml` and per-workspace settings.

## Run the CLI

The `ragbot` CLI is available inside the running container:

```bash
docker compose exec ragbot ragbot chat -profile personal -p "What are my travel preferences?"
docker compose exec ragbot ragbot skills list
docker compose exec ragbot ragbot db status
```

To install the CLI on the host instead (requires Python 3.14+):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ragbot --help
```

## Configure API keys

Drop your provider keys into `~/.synthesis/keys.yaml` (shared across the synthesis-engineering tools) or set them in `.env` for Docker-only use. See [CONFIGURE.md](CONFIGURE.md) for the schema, per-workspace overrides, and the legacy `~/.config/ragbot/` fallback.

## Update Ragbot

```bash
cd ragbot
git pull
docker compose pull
docker compose up -d --build
```

## Troubleshooting

- **Port conflicts.** If `3000` or `8000` is already in use, edit the published ports in `docker-compose.override.yml`.
- **Ollama not reachable.** Inside the container, host Ollama is `host.docker.internal:11434` on macOS / Windows. On Linux, set `OLLAMA_API_BASE=http://<host-ip>:11434` in `.env`.
- **No models in the picker.** Confirm at least one provider key in `.env` or `~/.synthesis/keys.yaml` and restart the stack.

For deployment guidance and production hardening, see [README-DOCKER.md](README-DOCKER.md).
