FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

WORKDIR /app

# Install PyTorch with CUDA 11.8
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install project deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY src/ src/
COPY api_ml.py .
COPY frontend/ frontend/

# Create dirs
RUN mkdir -p runs/inputs runs/outputs runs/jobs models

EXPOSE 8000

CMD ["uvicorn", "api_ml:app", "--host", "0.0.0.0", "--port", "8000"]
