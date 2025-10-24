# ============================================================
# Base image: CUDA 12.1 + Ubuntu 24.04
# ============================================================
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

LABEL maintainer="yourname@example.com"
LABEL description="Garment Folding dual-arm manipulation environment"

ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------
# 1. Install system dependencies and Python 3.12
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl python3 python3-venv python3-pip build-essential cmake \
    libusb-1.0-0-dev libgl1-mesa-glx udev && \
    rm -rf /var/lib/apt/lists/*


RUN ln -sf /usr/bin/python3.10 /usr/bin/python3

# ------------------------------------------------------------
# 2. Clone your repository (develop branch)
# ------------------------------------------------------------
WORKDIR /workspace
RUN git clone https://github.com/halid1020/agent-arena-v0.git && \
    cd agent-arena-v0 && git checkout develop

WORKDIR /workspace/agent-arena-v0

# ------------------------------------------------------------
# 3. Install uv (modern, fast installer for pyproject.toml)
# ------------------------------------------------------------
RUN pip install --upgrade pip && pip install uv

# ------------------------------------------------------------
# 4. Create a virtual environment and install project
# ------------------------------------------------------------
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install main project + optional [torch] extras
RUN uv pip install -e ".[torch]"

# ------------------------------------------------------------
# 5. Environment variables
# ------------------------------------------------------------
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ------------------------------------------------------------
# 6. Default command
# ------------------------------------------------------------
CMD ["/bin/bash"]
