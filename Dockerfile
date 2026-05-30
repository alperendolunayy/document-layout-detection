FROM python:3.12-slim

WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# install poetry
RUN pip install --no-cache-dir poetry

# copy dependency files first for better caching
COPY pyproject.toml poetry.lock* ./

# install dependencies without dev packages
RUN poetry config virtualenvs.create false && \
    poetry install --no-root --no-interaction --no-ansi --without dev

# copy project files
COPY . .

# install the project itself
RUN poetry install --no-interaction --no-ansi --without dev

# expose ports for gradio and fastapi
EXPOSE 8080 8000

# default command: launch gradio demo
CMD ["python", "-m", "docuvision.commands", "demo"]
