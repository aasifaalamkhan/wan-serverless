FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git wget ffmpeg libgl1-mesa-dev libglib2.0-0 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variable for HuggingFace large file transfers (optional)
ENV HF_HUB_ENABLE_HF_TRANSFER=1

RUN mkdir -p /app

# --- Install FlashAttention from prebuilt wheel (compatible with torch 2.4 and python 3.11) ---
RUN pip install \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu124torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Copy and install Python dependencies (excluding flash_attn to avoid rebuild)
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy application files
COPY ./app /app
COPY ./wan /wan

WORKDIR /app
CMD ["python3", "-u", "/app/rp_handler.py"]
