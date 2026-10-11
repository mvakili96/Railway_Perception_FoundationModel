# Build context: containers/qwen (not the repo, datasets, or model weights).
# Verified official linux/amd64 image; retain its Python environment.
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel@sha256:5a046e4e3364b063a17854387b8820ad3f42ed197a089196bce8f2bd68f275a8

ENV DEBIAN_FRONTEND=noninteractive \
    DS_BUILD_OPS=0 \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ninja-build git ca-certificates libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/qwen/requirements.txt

# The base already supplies CUDA torch/torchvision. Their requirements pins
# prevent the additional packages from silently changing that stack.
# No GPU is needed to build: optional DeepSpeed ops can JIT on HPC later.
RUN /opt/conda/bin/python -m pip install --no-cache-dir --upgrade \
        pip setuptools wheel packaging ninja \
    && /opt/conda/bin/python -m pip install --no-cache-dir --no-build-isolation \
        -r /opt/qwen/requirements.txt \
    && /opt/conda/bin/python -m pip check

# CPU-only import/version gate. CUDA execution is a separate HPC smoke test.
RUN /opt/conda/bin/python -c 'import torch, torchvision, transformers, peft, deepspeed, cv2; from pycocotools.coco import COCO; from pycocotools import mask; from transformers import Qwen3VLForConditionalGeneration, AutoProcessor; assert torch.__version__ == "2.7.1+cu118"; assert torchvision.__version__ == "0.22.1+cu118"; assert torch.version.cuda == "11.8"; assert transformers.__version__ == "4.57.6"; assert peft.__version__ == "0.17.1"; assert deepspeed.__version__ == "0.17.6"' \
    && /opt/conda/bin/python -m pip freeze --all > /opt/qwen/installed-packages.txt \
    && /opt/conda/bin/python --version > /opt/qwen/python-version.txt

LABEL org.opencontainers.image.description="Native Qwen3-VL environment; independent of legacy LISA"
WORKDIR /workspace
