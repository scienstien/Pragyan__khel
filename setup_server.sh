#!/bin/bash
# ============================================
# SmartFocus — EC2 Server Setup Script
# Run this on a fresh Ubuntu 22.04 GPU instance
# (e.g. AWS Deep Learning AMI on g4dn.xlarge)
# ============================================
set -e

echo "=== SmartFocus Server Setup ==="

# 1. Install Docker (if not present)
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    sudo usermod -aG docker $USER
    echo "Docker installed."
fi

# 2. Install NVIDIA Container Toolkit (if not present)
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    echo "Installing NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo "NVIDIA Container Toolkit installed."
fi

# 3. Create models directory and download weights
mkdir -p models runs/inputs runs/outputs runs/jobs
if [ ! -f models/yolov8n-seg.pt ]; then
    echo "Downloading YOLOv8n-seg weights..."
    pip install ultralytics -q 2>/dev/null || true
    python3 -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt')" 2>/dev/null || true
    # Move if downloaded to current dir
    [ -f yolov8n-seg.pt ] && mv yolov8n-seg.pt models/
    echo "Weights ready."
fi

# 4. Build and run
echo "Building Docker image..."
sudo docker compose build

echo "Starting SmartFocus..."
sudo docker compose up -d

echo ""
echo "============================================"
echo "  SmartFocus is running!"
echo "  API:      http://$(curl -s ifconfig.me):8000"
echo "  Frontend: http://$(curl -s ifconfig.me):8000/"
echo "  Health:   http://$(curl -s ifconfig.me):8000/health"
echo "============================================"
echo ""
echo "Useful commands:"
echo "  sudo docker compose logs -f    # view logs"
echo "  sudo docker compose down       # stop"
echo "  sudo docker compose up -d      # start"
echo "  nvidia-smi                     # check GPU"
