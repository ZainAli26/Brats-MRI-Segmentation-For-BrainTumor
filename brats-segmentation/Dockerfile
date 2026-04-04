# ==============================================================================
# BraTS MRI Segmentation — GPU Training Container
#
# Base: PyTorch 2.4 + CUDA 12.4 (compatible with host driver 555.x / CUDA 12.5)
# Includes: MONAI, nnU-Net v2, all experiment configs
#
# Build:
#   docker build -t brats-seg .
#
# Run:
#   docker run --gpus all -v /path/to/data:/data -v /path/to/runs:/app/runs \
#       brats-seg python train.py --config experiments/exp01_segresnet_baseline.yaml
# ==============================================================================

FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies for nibabel, matplotlib, scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (cached layer — only rebuilds if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY src/ src/
COPY configs/ configs/
COPY experiments/ experiments/
COPY nnunet_native/ nnunet_native/
COPY train.py evaluate.py analyze_failures.py ./

# Create output directories
RUN mkdir -p runs output

# nnU-Net environment variables (overridden at runtime via -v mounts)
ENV nnUNet_raw=/app/nnunet_data/nnUNet_raw
ENV nnUNet_preprocessed=/app/nnunet_data/nnUNet_preprocessed
ENV nnUNet_results=/app/nnunet_data/nnUNet_results

# Default: show help
CMD ["python", "train.py", "--help"]
