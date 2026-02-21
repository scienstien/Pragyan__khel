🧠 AI-Based Smart Autofocus & Dynamic Subject Tracking System

Faustian_for{loop}

🚀 Overview

This project implements an AI-powered Smart Autofocus & Dynamic Subject Tracking System that allows a user to:

Click on any object in a video

Track the selected subject across frames

Keep the selected subject sharp

Apply background blur to all other objects

Switch focus instantly by clicking another subject

Render a cinematic output video

The system combines modern computer vision techniques such as object detection, multi-object tracking, and segmentation-based compositing to deliver a real-time interactive experience.

🎯 Problem Statement
Traditional autofocus systems focus only on faces or predefined objects.
Our goal was to build a system that:
_________________________________________
1. Allows arbitrary object selection

2. Maintains subject identity across frames

3. Handles occlusion and motion

4. Applies high-quality segmentation-based blur

5. Runs on-device using GPU acceleration
_________________________________________________
🏗️ System Architecture
1️⃣ Vision Layer (AI Engine)

.YOLOv8 (Segmentation variant) for object detection and mask generation

.ByteTrack for persistent multi-object tracking

.Pixel-level segmentation for accurate subject isolation

.Selective Gaussian blur compositing for cinematic focus

2️⃣ Backend Layer (FastAPI)

Handles video uploads

Manages per-video tracking state

Exposes REST API endpoints

Uses GPU acceleration (RTX 4060) ##NOTE : WINDOWS   FIREWALL MIGHT INTERFARE DUE TO IT BEING A PRIVATE NETWORK

3️⃣ Frontend Layer (Simple Html with websockets)

.Upload interface

.Frame preview rendering

.Click-to-select interaction

.Play/Pause controls

.Render & download functionality

🧠 Core Technical Concepts
____________________________________________________
🔹 Object Detection (YOLOv8)

A one-stage real-time detector that predicts bounding boxes and segmentation masks in a single forward pass.

🔹 Multi-Object Tracking (ByteTrack)

Maintains consistent identity (track_id) across frames, ensuring that once a user selects an object, the same physical object continues to be tracked.

🔹 Click-to-Identity Lock

When the user clicks:

We determine which bounding box contains the click

We lock its track_id

All future frames follow this identity

🔹 Segmentation-Based Blur

For each frame:

1. Extract subject mask
    |
    |
2. Blur entire frame
    |
    |
3. Composite sharp subject over blurred background
     |
     |
4. Feather edges for smooth transitions
____________________________________________________
✨ Key Features

..Real-time object selection

..Persistent identity tracking

..Segmentation-based blur (not bounding-box blur)

..Instant focus switching

..GPU acceleration

..Modular architecture

..Scalable backend API

🎥 Live Camera Capability (V2)

Our system architecture supports real-time live camera blur.

We have already implemented the capability to:

Capture live webcam feed

Perform detection + tracking in real time

Apply segmentation-based background blur

However, in this hackathon version, we focused on uploaded video processing for stability and evaluation clarity.
_________________________________________________________________________________________________
In Version 2, we plan to:

Integrate live webcam streaming into the frontend

Enable real-time autofocus in browser

Support WebRTC-based streaming

Deploy optimized streaming endpoints
________________________________________________________________________________
🖥️ Hardware Requirements

NVIDIA GPU recommended (RTX 4060 used in development)

CUDA-compatible PyTorch

8GB+ RAM recommended

📦 Installation & Setup Guide
1️⃣ Clone the Repository
```  bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
2️⃣ Create Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
Mac/Linux
python3 -m venv venv
source venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt

If running on GPU, ensure CUDA-enabled PyTorch is installed:

Visit:
https://pytorch.org/get-started/locally/

4️⃣ Run Backend (FastAPI)
uvicorn api_ml:app --host 0.0.0.0 --port 8000
````
Check:

```
http://localhost:8000/health
http://localhost:8000/docs
```
5️⃣ Run Frontend (Next.js)

Navigate to frontend folder:

```
cd Pragyaan_khel/frontend
```
Open:

http://localhost:3000
🌐 Network Setup (If Using Multiple Laptops)

If frontend and backend are on different machines:

Find backend laptop IP:

ipconfig

Use:

http://<your-ip>:8000

instead of localhost

📡 API Endpoints
```
Upload Video
POST /upload
Get Preview Frame
GET /frame
Select Target
POST /select
Reset Focus
POST /reset
Render Output
POST /render
Download Final Video
GET /download
````
🧪 Running Locally Without Frontend

You can test the model using:

python main.py

This opens a local OpenCV window for:

Click-to-select

Frame stepping

Blur preview

⚙️ Project Structure
src/
  focus_engine.py
api_ml.py
main.py
models/
runs/
TeamName_NextGenHackathon/
🧩 Challenges Faced

Click coordinate mismatch due to resizing

Frame seeking latency in OpenCV

Maintaining tracking identity across occlusion

Optimizing blur performance for real-time preview

🌍 Real-World Applications

1.Smart camera autofocus systems

2.Video conferencing tools

3.Cinematic video editing

4.Content creation platforms

5.Surveillance analytics

6.Sports tracking systems like Cricket and F1

🚀 Future Improvements
>>>>>__________________________________<<<<
Live webcam integration

WebRTC streaming

Depth-aware blur

Mobile deployment

Edge device optimization

Cloud deployment

Due to lack of infra we couldnt deploy it to aws this time it was asking 50GB worth of compute
>>>>>__________________________________<<<<

🏁 Conclusion
______________________________________
This project demonstrates how modern computer vision techniques can be integrated into a scalable, interactive, and real-world application for dynamic autofocus and subject tracking.

Our modular design ensures extensibility for live camera streaming, edge deployment, and production-scale integration.
