# Dockerfile for Ragbot.AI
# FastAPI backend for CLI and React web interface

# Python 3.14 is the newest release every dependency ships wheels for, on
# both linux/amd64 and linux/arm64 (checked 2026-11-06; 3.15 still lacks
# wheels for PyYAML, fastuuid and orjson). With wheels for everything, the
# image needs no compiler.
FROM python:3.14-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

FROM base AS dependencies

COPY requirements.txt .
# torch comes from PyTorch's CPU index. PyPI's Linux wheels are the CUDA
# builds, which put 3.3 GB of nvidia-* packages and 0.9 GB of Triton into an
# image that has no GPU to use them: Docker Desktop on a Mac passes none
# through, and docker-compose.yml reserves none. On 2026-11-06 that payload
# made the image 6.8 GB and filled Docker's disk mid-build.
#
# --no-deps keeps torch's dependencies off PyTorch's index, whose mirrors of
# them can lag PyPI: the next install resolves them from PyPI, because pip
# resolves an installed package's dependencies when a requirement
# (sentence-transformers) pulls it in. pip check then fails the build if any
# installed package's requirement is unmet.
RUN pip install --no-cache-dir --no-deps --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements.txt \
    && pip check

FROM base AS final

COPY --from=dependencies /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

COPY src/ ./src/
COPY engines.yaml .
COPY VERSION .
COPY ragbot ragbot_api ./
# Bundled demo workspace + demo skill pack used by RAGBOT_DEMO=1.
# Discovery short-circuits to this content when demo mode is active.
COPY demo/ ./demo/

RUN chmod +x ragbot ragbot_api

RUN mkdir -p /root/.local/share/ragbot/sessions && \
    mkdir -p /root/.synthesis && \
    mkdir -p /app/ragbot-data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
