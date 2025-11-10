# Prep Pal Backend

This directory contains the backend services that power the Prep Pal tutoring experience:

- `api/` — FastAPI service that ingests study guides, stores session metadata in Firestore, and issues LiveKit credentials
- `agent/` — LiveKit Agent that streams the Hedra avatar, orchestrates Gemini reasoning, and drives flash cards and quizzes

Both services depend on Google Cloud and LiveKit resources. Keep credentials in local environment files and never commit real keys to the repository.

## Prerequisites
- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/) (recommended) or `pip`/`venv` for dependency management
- Google Cloud project with Firestore and Vertex AI (Gemini) enabled
- LiveKit server (Cloud or self-hosted) with an API key and secret
- Hedra avatar API access and ElevenLabs API key (or adjust the agent to your providers)

## Environment Variables
Create a `.env` file in `prep-pal-backend/` with your own values:

```bash
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1

LIVEKIT_URL=wss://your-livekit-host
LIVEKIT_API_KEY=your-livekit-api-key
LIVEKIT_API_SECRET=your-livekit-api-secret

HEDRA_API_KEY=your-hedra-api-key
ELEVEN_API_KEY=your-elevenlabs-api-key
```

> **Note:** If you cloned this codebase from a private repo, scrub any existing `.env` file for real secrets before committing. Publish placeholders only.

Both the API and the agent load this `.env` via `python-dotenv`.

## Project Layout
```
prep-pal-backend/
├── api/           # FastAPI session configuration service
├── agent/         # LiveKit avatar agent implementation
└── README.md      # You are here
```

Each subdirectory ships its own `pyproject.toml`; run dependency commands from the relevant folder.

## Install Dependencies
Examples below use `uv`, but feel free to substitute `pip`.

```bash
# Session API
cd prep-pal-backend/api
uv sync

# Avatar agent
cd ../agent
uv sync
```

If you prefer `pip`, create a virtual environment (`python -m venv .venv && source .venv/bin/activate`) and run `pip install -e .`.

## Run the Session API
```bash
cd prep-pal-backend/api
uv run uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

The frontend expects the `/session-config` endpoint exposed over HTTPS in production. Tighten CORS before making the service public.

## Run the Avatar Agent
```bash
cd prep-pal-backend/agent
uv run python avatar.py
```

The agent connects to the LiveKit room created by the API, publishes the Hedra avatar feed, and handles flash card or quiz RPCs. Verify your LiveKit and Google credentials before launching.

## Development Notes
- **Firestore** — update `SESSION_COLLECTION` in `api/app.py` or `agent/avatar.py` to match your data layout.
- **Gemini model** — adjust the `model` argument in `api/app.py` (`generate_topic_prompt`) to align with your Vertex AI region/quota.
- **Logging** — both services use Python `logging`; set `LOGLEVEL=DEBUG` or configure handlers as needed.
- **Testing** — add unit/integration tests and run them via `uv run pytest` (tests are not bundled yet).

## Deployment Checklist
- Store secrets in your hosting provider’s secret manager instead of `.env` files.
- Place the FastAPI service behind HTTPS and enforce trusted origins.
- Use LiveKit’s worker orchestration or a process supervisor to keep the agent running.
- Rotate API keys regularly and monitor usage.

## Contributing
Open an issue or pull request with a clear description and testing notes. Before submitting:

```bash
cd prep-pal-backend/api && uv run ruff check .
cd ../agent && uv run ruff check .
```

Linting should pass in both subprojects to keep CI healthy.
