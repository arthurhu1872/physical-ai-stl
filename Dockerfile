# syntax=docker/dockerfile:1.7-labs
#
# physical-ai-stl — reproducible, CPU-first dev/CI image with optional extras.
# Defaults meet course needs: tiny, deterministic, and able to run tests headlessly.
# Heavy stacks (Neuromancer, TorchPhysics, MoonLight/STREL, PhysicsNeMo) are opt‑in.
#
# Usage (CPU, minimal):
#   docker build -t pai-stl .
#   docker run --rm -it pai-stl
#
# With extras (incl. STL/STREL + physics‑ML toolkits) and Java for MoonLight:
#   docker build --build-arg WITH_EXTRAS=1 --build-arg WITH_JAVA=1 -t pai-stl:full .
#
# With CUDA wheels for torch (requires NVIDIA Container Toolkit on host):
#   docker build --build-arg WITH_EXTRAS=1 --build-arg TORCH_CHANNEL=cu121 -t pai-stl:cuda .
#
# Align default Python with CI while keeping it configurable.
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS runtime

# Use bash + strict mode for robustness
SHELL ["/bin/bash", "-Eeuo", "pipefail", "-c"]

# OCI labels for provenance
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="physical-ai-stl" \
      org.opencontainers.image.description="Reproducible environment for STL/STREL monitoring with physics-based ML (Neuromancer, PhysicsNeMo, TorchPhysics). CPU-first; extras opt-in." \
      org.opencontainers.image.url="https://github.com/arthurhu1872/physical-ai-stl" \
      org.opencontainers.image.source="https://github.com/arthurhu1872/physical-ai-stl" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}"

# ---- deterministic, quiet Python ------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TZ=UTC

# ---- build-time feature toggles -------------------------------------------------
# 0 = off, 1 = on
ARG WITH_EXTRAS=0          # heavy research stacks (STL/STREL + physics‑ML)
ARG WITH_DEV=0             # dev tools (linters, pytest, etc.)
ARG WITH_JAVA=0            # Java runtime for MoonLight (STREL) Python wrapper
ARG TORCH_CHANNEL=cpu      # cpu | cu121 | cu118 (only used when WITH_EXTRAS=1)

# Some extras (e.g., PhysicsNeMo) are hosted on NVIDIA's index.
ENV PIP_EXTRA_INDEX_URL="https://pypi.nvidia.com/simple"

# System deps kept lean by default; extras are gated below.
# Use BuildKit cache mounts to speed up iterative builds.
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    set -eux; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get update; \
    # Core utilities for building common wheels (cmake for RTAMT C++ backend) \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        curl \
        build-essential \
        pkg-config \
        cmake; \
    # Optional: Java for MoonLight/STREL and MONA for SpaTiaL (only when requested) \
    if [ "$WITH_JAVA" = "1" ] || [ "$WITH_EXTRAS" = "1" ]; then \
        apt-get install -y --no-install-recommends \
            openjdk-21-jre-headless \
            mona \
            libboost-all-dev; \
    fi; \
    rm -rf /var/lib/apt/lists/*

# If Java installed, expose JAVA_HOME for Python wrappers.
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Create a non-root user for nicer dev UX
ARG USER=app
ARG UID=1000
ARG GID=1000
RUN set -eux; \
    groupadd -g "${GID}" "${USER}"; \
    useradd -m -u "${UID}" -g "${GID}" -s /bin/bash "${USER}"
WORKDIR /workspace

# Upgrade pip/setuptools/wheel once (kept cached across code changes)
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel

# Copy only files needed to resolve Python deps first for better layer caching
COPY pyproject.toml requirements.txt requirements-extra.txt requirements-dev.txt ./

# Minimal, fast runtime deps (numpy only)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Optional heavy stacks (Neuromancer, TorchPhysics, RTAMT, MoonLight, PhysicsNeMo)
# Torch is installed explicitly from the official index first to avoid fetching CUDA
# unless TORCH_CHANNEL is set to a CUDA variant.
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$WITH_EXTRAS" = "1" ]; then \
        case "${TORCH_CHANNEL}" in \
          cpu)   pip install --index-url https://download.pytorch.org/whl/cpu torch;; \
          cu121) pip install --index-url https://download.pytorch.org/whl/cu121 torch;; \
          cu118) pip install --index-url https://download.pytorch.org/whl/cu118 torch;; \
          *)     pip install torch;; \
        esac; \
        # Install everything *except* moonlight first (to handle Java availability cleanly) \
        grep -vE '^[[:space:]]*#|^[[:space:]]*$' requirements-extra.txt | grep -vi '^moonlight' > /tmp/req-extra.nomoonlight.txt; \
        pip install -r /tmp/req-extra.nomoonlight.txt; \
        # Try PyPI first (recent releases require Java 21+), then fall back to upstream GitHub. \
        (pip install 'moonlight' || pip install 'git+https://github.com/MoonLightSuite/moonlight') || true; \
    fi

# Dev tooling (pytest, ruff, mypy, etc.) — mirrors CI locally
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$WITH_DEV" = "1" ]; then \
        pip install -r requirements-dev.txt; \
    fi

# Bring in the rest of the repository and install the package in editable mode
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY Makefile README.md CITATION.cff LICENSE ./

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e .

# Default command mirrors CI: run tests if pytest is available; otherwise no-op with hint.
USER ${USER}
CMD ["bash", "-lc", "if command -v pytest >/dev/null 2>&1; then python -m pytest -q; else echo 'pytest not installed (build with --build-arg WITH_DEV=1 to run tests).'; fi"]
