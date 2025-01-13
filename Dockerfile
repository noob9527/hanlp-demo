# There isn't much difference in size when use uv image
# https://github.com/astral-sh/uv-docker-example/blob/main/Dockerfile
# FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
COPY ./pyproject.toml /app/pyproject.toml
COPY ./uv.lock /app/uv.lock
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Set HANLP_HOME environment variable
ENV HANLP_HOME=/app/hanlp

# Create directory for HanLP models
RUN mkdir -p ${HANLP_HOME}

# Pre-download all required models
RUN uv run python -c 'import hanlp; \
    hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH); \
    hanlp.load(hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH); \
    hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL); \
    hanlp.load(hanlp.pretrained.pos.PKU_POS_ELECTRA_SMALL)'


COPY ./src /app/src

# https://docs.astral.sh/uv/guides/integration/docker/#using-the-environment
# Set PATH and make sure it persists
ENV PATH="/app/.venv/bin:$PATH"
# RUN echo "PATH=/app/.venv/bin:$PATH" >> /etc/environment
# RUN echo 'export PATH="/app/.venv/bin:$PATH"' >> /root/.bashrc

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

# CMD ["uv", "run", "src/server.py", "--port", "80"]
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "80"]
