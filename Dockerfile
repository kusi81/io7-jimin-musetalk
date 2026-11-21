# RunPod Serverless MuseTalk 1.5 Worker — IO7 Sales (김지민 과장 1080p 실시간 립싱크)
# Recommended GPU: NVIDIA L4 (24GB, ~$0.013/min Flex) or RTX 4090 (24GB, ~$0.022/min Flex)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/kusi81/io7-jimin-musetalk" \
      org.opencontainers.image.description="IO7 Sales Kim Ji-min Gwazang MuseTalk 1.5 Real-Time Lip-Sync Serverless Worker"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/models/hf_cache \
    MUSETALK_DIR=/app/MuseTalk \
    PREBAKED_DIR=/app/prebaked \
    ASSETS_DIR=/app/assets

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev \
    git ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libsndfile1 \
    curl wget ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# 1. Install PyTorch (CUDA 12.1) & lean MuseTalk 1.5 runtime dependencies
RUN pip install --upgrade pip && \
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
        --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# 2. Clone MuseTalk 1.5 (official TMElyralab repo)
RUN git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git /app/MuseTalk

# 3. Pre-download MuseTalk v1.5 UNet, SD-VAE-FT-MSE, and Whisper-tiny weights into image
#    Eliminates runtime download during RunPod Serverless cold start (FlashBoot-ready!)
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('TMElyralab/MuseTalk', local_dir='/app/MuseTalk/models', allow_patterns=['musetalkV15/*']); \
snapshot_download('stabilityai/sd-vae-ft-mse', local_dir='/app/MuseTalk/models/sd-vae'); \
snapshot_download('openai/whisper-tiny', local_dir='/app/MuseTalk/models/whisper')"

# 4. Copy Kim Ji-min Gwazang 1080p Hub clips & worker scripts
COPY assets/ /app/assets/
COPY prebake.py /app/prebake.py
COPY handler.py /app/handler.py

# 5. Pre-bake facial bounding boxes & 256x256 crops at build time
RUN mkdir -p /app/prebaked && python /app/prebake.py --assets /app/assets --out /app/prebaked --allow-cpu-fallback

CMD ["python", "-u", "/app/handler.py"]
