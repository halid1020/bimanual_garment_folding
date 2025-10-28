# Build: docker build -t garment_folding:latest .
# RUN:  xhost +local:root
# sudo docker run -it     --privileged     --runtime=nvidia --gpus all     -e DISPLAY=$DISPLAY     -v /tmp/.X11-unix:/tmp/.X11-unix     --device=/dev/bus/usb     -v /dev/bus/usb:/dev/bus/usb     -v ~/Projects/bimanual_garment_folding:/workspace/bimanual_garment_folding     garment_folding:latest

# ============================================================
# Base image: CUDA 12.1 + Ubuntu 24.04
# ============================================================

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

LABEL maintainer="halidkadi.robot@gmail.com"
LABEL description="Garment Folding dual-arm manipulation environment"

ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------
# 1. Install system dependencies and Python 3.12
# ------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl python3 python3-venv python3-pip build-essential cmake \
    libusb-1.0-0-dev libgl1-mesa-glx udev \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxkbcommon-x11-0 \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-shm0 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb1 \
    libsm6 libxext6 libxrender1 libfontconfig1 libfreetype6 && \
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

RUN pip install pycurl
RUN pip install git+https://github.com/facebookresearch/segment-anything.git

RUN mkdir -p /workspace/bimanual_garment_folding/models && \
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
    -O /workspace/bimanual_garment_folding/models/sam_vit_h_4b8939.pth

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
