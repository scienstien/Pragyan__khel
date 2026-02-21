# SmartFocus — AWS GPU Deployment Guide

## Quick Start (3 steps)

### Step 1: Launch EC2 GPU Instance

1. Go to **AWS Console → EC2 → Launch Instance**
2. Choose these settings:
   - **Name**: `SmartFocus`
   - **AMI**: Search for `Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)`
     - *(This AMI has NVIDIA drivers pre-installed, saves ~20 min of setup)*
   - **Instance type**: `g4dn.xlarge` (~$0.53/hr, 1× T4 GPU, 4 vCPU, 16 GB RAM)
   - **Key pair**: Create or select one (you'll need this to SSH)
   - **Security Group**: Allow these inbound rules:
     - SSH (port 22) — from your IP
     - Custom TCP (port 8000) — from anywhere (0.0.0.0/0)
   - **Storage**: 50 GB gp3 (default 8GB won't be enough for Docker + model)
3. Click **Launch Instance**
4. Note the **Public IPv4 address** once running

### Step 2: Upload Code to EC2

From your local machine (Windows/Mac), open a terminal:

```bash
# Replace with YOUR key file path and EC2 public IP
KEY=~/Downloads/your-key.pem
EC2=ubuntu@<YOUR_EC2_PUBLIC_IP>

# Fix key permissions (needed on Mac/Linux, skip on Windows)
chmod 400 $KEY

# Copy project to EC2
scp -i $KEY -r ./Pragyaan__hackathon $EC2:~/smartfocus
```

**On Windows (PowerShell)**:
```powershell
$KEY = "C:\Users\you\Downloads\your-key.pem"
$EC2 = "ubuntu@<YOUR_EC2_PUBLIC_IP>"

scp -i $KEY -r .\Pragyaan__hackathon ${EC2}:~/smartfocus
```

### Step 3: SSH In and Run Setup

```bash
# SSH into EC2
ssh -i $KEY $EC2

# Run setup (installs Docker, downloads model, starts server)
cd ~/smartfocus
bash setup_server.sh
```

That's it! The script will print the URL when done.

---

## Access Your App

- **Frontend**: `http://<EC2_IP>:8000/`
- **API Health**: `http://<EC2_IP>:8000/health`
- **GPU check**: Inside EC2, run `nvidia-smi`

## Useful Commands (on EC2)

```bash
# View logs
sudo docker compose logs -f

# Restart
sudo docker compose restart

# Stop
sudo docker compose down

# Rebuild after code changes
sudo docker compose build && sudo docker compose up -d
```

## Cost Tips

- **Don't forget to stop the instance** when not using it (EC2 → Instance → Stop)
- Stopped instances have no compute charges (only EBS storage ~$4/mo)
- For long-term use, consider **Spot instances** (~70% cheaper)
