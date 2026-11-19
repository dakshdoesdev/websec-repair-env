ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app/env

COPY . /app/env

RUN python -m pip install --no-cache-dir "openenv-core[core]>=0.2.1" && \
    python -m pip install --no-cache-dir --no-deps -e /app/env

ENV PYTHONPATH="/app/env:${PYTHONPATH}"

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
