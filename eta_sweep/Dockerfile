# Reproducible CUDA environment for the η-sweep NeurIPS submission.
#
# Build:
#   docker build -t eta-sweep:latest -f eta_sweep/Dockerfile .
#
# Run a single cell deterministically:
#   docker run --rm --gpus all \
#     -v $PWD/eta_sweep/results:/workspace/eta_sweep/results \
#     eta-sweep:latest \
#     python eta_sweep/run_single.py --eta 3e-4 --k 36 --seed 0 \
#       --min-steps 20000 --stuck-patience 5000 --max-steps 100000
#
# The Dockerfile installs a pinned torch CUDA wheel separately from the
# rest of the dependencies so that rebuilds don't redownload torch.

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUBLAS_WORKSPACE_CONFIG=:16:8 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3-pip \
      build-essential git \
      ca-certificates \
      && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1

RUN python -m pip install --upgrade pip==24.2

# ---- Torch CUDA wheel (pinned to the exact version used in the MPS env) -----
# torch 2.6.0 + CUDA 12.4 extra-index-url.
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
      torch==2.6.0

# ---- Remaining pinned Python deps -------------------------------------------
WORKDIR /workspace
COPY eta_sweep/requirements-pinned.txt /workspace/eta_sweep/requirements-pinned.txt
RUN pip install -r /workspace/eta_sweep/requirements-pinned.txt

# ---- Project source ---------------------------------------------------------
# Copy only what's needed to run the sweep + analysis.
COPY src/   /workspace/src/
COPY configs/ /workspace/configs/
COPY eta_sweep/ /workspace/eta_sweep/

WORKDIR /workspace
ENV PYTHONPATH=/workspace

# Smoke test at build time so broken builds fail fast.
RUN python -c "import torch, transformer_lens; \
    print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); \
    print('transformer_lens OK')"

# Default entrypoint runs the multiseed sweep; override with `docker run ... python <other>`.
CMD ["python", "eta_sweep/multiseed_sweep.py", "--workers", "4"]
