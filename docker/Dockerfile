FROM hub.rat.dev/nvidia/cuda:12.8.0-devel-ubuntu22.04

USER root

# Use local mirror and install required tools + Python
COPY docker/sources.list /etc/apt/sources.list
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git sudo \
    python3 python3-venv python3-dev \
    build-essential pkg-config libevdev-dev libgl1-mesa-glx libglib2.0-0 vim \
  && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN if ! getent group user >/dev/null; then groupadd -g 1000 user || groupadd user; fi
RUN if ! getent passwd user >/dev/null; then useradd -create-home -u 1000 -g user -m user || useradd -create-home -g user -m user; fi
RUN usermod -aG sudo user && echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# pip mirror for the user
RUN mkdir -p /home/user/.pip
COPY --chown=user docker/pip.conf /home/user/.pip/pip.conf

# Put project at /workspace (same path you'll mount at runtime)
WORKDIR /workspace
COPY --chown=user . .
## Ensure the project root itself is writable by the non-root user
RUN chown user:user /workspace

## Pre-create venv directory with correct ownership so the non-root user can write to /opt
RUN mkdir -p /opt/lerobot && chown -R user:user /opt/lerobot

## Install uv globally so it's always on PATH
ENV UV_INSTALL_DIR=/usr/local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && uv --version
ENV UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

USER user

# Create venv with system Python 3.10
RUN python3 -m venv /opt/lerobot
ENV VIRTUAL_ENV=/opt/lerobot
ENV PATH="/opt/lerobot/bin:$PATH"

# Install the root project and the detr submodule in editable mode
RUN uv pip install -e .
RUN uv pip install scipy glog dataset

WORKDIR /workspace
