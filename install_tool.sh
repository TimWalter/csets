#!/bin/bash

# install_tool.sh — run once on the worker to install csets (CORA-COMP interface).
#
# The platform clones this repository into the Docker base image named on the submission
# form and runs this script from it. It installs uv and syncs the locked environment
# (JAX + Moreau with CUDA 13 wheels) into ./.venv; run_instance.sh uses that interpreter.
#
# Argument:
# - $1: interface version string, e.g. "v1"

set -euo pipefail

VERSION="${1:-v1}"
echo "Installing csets (interface $VERSION)"

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# uv downloads a matching Python (>=3.14, see pyproject.toml) if the image has none.
uv sync --frozen --no-dev

.venv/bin/python -W ignore -c "import jax, moreau; print('jax', jax.__version__, 'moreau', moreau.__version__, 'devices', jax.devices())"

# The worker, for choosing benchmark/config.env's CPU settings.
echo "--- worker: $(nproc) CPUs available to this process"
lscpu | grep -E "^(Model name|Socket|Core|Thread|CPU\(s\)|NUMA node\(s\))" || true
free -g | head -2 || true
nvidia-smi -L 2>/dev/null || echo "nvidia-smi: not present"
. ./benchmark/config.env; echo "benchmark/config.env: CSETS_CPU_DEVICES=$CSETS_CPU_DEVICES"
