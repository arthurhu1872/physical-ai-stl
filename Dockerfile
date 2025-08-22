# Tiny, CPU-only image that mirrors CI: Python 3.11 + pytest + minimal deps.
# Optional build args allow you to pull in extras or Java 21 for MoonLight.

# syntax=docker/dockerfile:1
FROM python:3.11-slim

# --- sane defaults -----------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PIP_NO_CACHE_DIR=1

# Toggle extras (heavy experiment deps) and Java support for MoonLight STREL.
ARG WITH_EXTRAS=0
ARG WITH_JAVA=0

# Base OS deps: compilers for native wheels, git for editable installs.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential git ca-certificates && \
    if [ "$WITH_JAVA" = "1" ]; then \
        apt-get install -y --no-install-recommends openjdk-21-jre-headless; \
    fi && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install minimal + dev test deps first (best layer caching).
COPY requirements.txt requirements-dev.txt ./
RUN python -m pip install --upgrade pip wheel && \
    pip install -r requirements.txt -r requirements-dev.txt

# Install the package itself (so `import physical_ai_stl` works).
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install -e .

# Optional: heavy experiment stacks (kept off by default).
# If you enable this, you likely also want torch CPU wheels explicitly.
COPY requirements-extra.txt requirements-extra.txt
RUN if [ "$WITH_EXTRAS" = "1" ]; then \
        pip install -r requirements-extra.txt || true; \
        pip install --index-url https://download.pytorch.org/whl/cpu torch || true; \
    fi

# Bring in scripts, configs, tests, etc., last (to avoid busting cache).
COPY scripts ./scripts
COPY configs ./configs
COPY tests ./tests
COPY Makefile ./Makefile

# Default: run tests (like CI). Override with `docker run ... bash` for dev.
CMD ["pytest", "-q"]
