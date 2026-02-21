# 🎯 SmartFocus — AI Video Focus Engine

SmartFocus is an AI-powered video processing tool that uses **YOLOv8 segmentation** with **ByteTrack** to let users click on any subject in a video and automatically blur the background while keeping the selected subject in sharp focus.

![SmartFocus UI](https://img.shields.io/badge/Status-Active-brightgreen) ![Python](https://img.shields.io/badge/Python-3.10+-blue) ![CUDA](https://img.shields.io/badge/CUDA-11.8-green) ![FastAPI](https://img.shields.io/badge/FastAPI-Latest-teal)

## ✨ Features

- **Click-to-Track** — Click on any person/object to lock focus
- **Real-time Preview** — WebSocket-powered live video stream with blur compositing
- **Instance Segmentation** — YOLOv8n-seg for pixel-accurate masks
- **Background Blur** — Gaussian blur with feathered mask edges
- **Export** — Render and download the focused video
- **GPU Accelerated** — CUDA support for fast inference

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML Model** | YOLOv8n-seg (Ultralytics) |
| **Tracking** | ByteTrack |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Vanilla HTML/CSS/JS |
| **Streaming** | WebSocket (binary JPEG frames) |
| **GPU** | NVIDIA CUDA 11.8 + PyTorch |
| **Container** | Docker + NVIDIA Container Toolkit |

## 📁 Project Structure

```
├── api_ml.py              # FastAPI server (upload, stream, select, render, download)
├── main.py                # Local OpenCV preview (desktop only)
├── src/
│   └── focus_engine.py    # Core engine — YOLO inference, tracking, mask compositing
├── frontend/
│   └── index.html         # Web UI (dark theme, WebSocket canvas)
├── models/                # YOLOv8 weights (auto-downloaded)
├── runs/                  # Video inputs, outputs, and job state
├── Dockerfile             # GPU-enabled container (CUDA 11.8)
├── docker-compose.yml     # One-command deploy with GPU passthrough
├── setup_server.sh        # EC2 server setup script
├── requirements.txt       # Python dependencies
└── DEPLOY.md              # AWS deployment guide
```

## 🚀 Quick Start

### Local (CPU — for testing)

```bash
pip install -r requirements.txt
pip install torch torchvision

# Start API server
uvicorn api_ml:app --host 0.0.0.0 --port 8000

# Open http://localhost:8000 in browser
```

### Docker (GPU)

```bash
# Build and run with GPU
docker compose up --build -d

# Check logs
docker compose logs -f
```

### AWS EC2 (Production GPU)

See [DEPLOY.md](DEPLOY.md) for full step-by-step instructions. TL;DR:

1. Launch **g4dn.xlarge** EC2 with Deep Learning AMI (Ubuntu 22.04)
2. Upload project: `scp -i key.pem -r . ubuntu@<IP>:~/smartfocus`
3. SSH in and run: `bash setup_server.sh`

App will be live at `http://<EC2_IP>:8000/`

## 🎮 Usage

1. **Upload** a video (MP4, AVI, MOV)
2. **Start Stream** to begin live preview
3. **Click** on the subject you want to focus on
4. The background blurs automatically while tracking the subject
5. **Render** the final video and **Download** it

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check + GPU info |
| `GET` | `/` | Frontend UI |
| `POST` | `/upload` | Upload video file |
| `WS` | `/ws/stream` | WebSocket video stream |
| `POST` | `/select` | Select target at (x, y) |
| `POST` | `/reset` | Reset target lock |
| `POST` | `/render` | Render focused video |
| `GET` | `/download` | Download rendered video |
| `POST` | `/close` | Close job |

## 🔧 Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `auto` | GPU device (`0`, `cpu`, or `auto`) |

## 📄 License

MIT


