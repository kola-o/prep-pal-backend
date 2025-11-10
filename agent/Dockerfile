# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install the project into `/app`
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy


# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=agent/uv.lock,target=uv.lock \
    --mount=type=bind,source=agent/pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Copy only the agent source code after dependencies are cached
COPY agent ./agent

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Pre-download any assets required by the agent to avoid fetching at runtime
#  RUN uv run agent/avatar.py download-files

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

# Run the LiveKit agent
CMD ["uv", "run", "agent/avatar.py", "start"]