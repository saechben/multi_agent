FROM python:3.13-slim

# Tools you usually want with Poetry
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install Poetry (isolated) – change version if you like
ENV POETRY_VERSION=1.8.3
RUN curl -sSL https://install.python-poetry.org | python - --version $POETRY_VERSION
ENV PATH="/root/.local/bin:$PATH"

# Make Poetry create .venv inside the project
ENV POETRY_VIRTUALENVS_IN_PROJECT=true