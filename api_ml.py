import os
import uuid
import shutil
from typing import Optional
import threading
import time
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
import asyncio
import struct
from fastapi import WebSocket, WebSocketDisconnect

from src.focus_engine import FocusEngine

# ----------------------------
# Global lock (thread-safe YOLO)
# ----------------------------
FRAME_LOCK = threading.Lock()

# ----------------------------
# Paths
# ----------------------------
RUNS_INPUT = os.path.join("runs", "inputs")
RUNS_OUTPUT = os.path.join("runs", "outputs")
RUNS_JOBS = os.path.join("runs", "jobs")
MODEL_PATH = os.path.join("models", "yolov8n-seg.pt")

os.makedirs(RUNS_INPUT, exist_ok=True)
os.makedirs(RUNS_OUTPUT, exist_ok=True)
os.makedirs(RUNS_JOBS, exist_ok=True)

# ----------------------------
# App
# ----------------------------
app = FastAPI(title="SmartFocus ML Service", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Engine (GPU)
# ----------------------------
engine = FocusEngine(
    model_path=MODEL_PATH,
    device="0",
    tracker="bytetrack.yaml",
    conf=0.25,
    iou=0.5,
    imgsz=640,
    job_root=RUNS_JOBS,
)

# ----------------------------
# Warmup (run once, after engine exists)
# ----------------------------
@app.on_event("startup")
def warmup():
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    try:
        # fuse once (avoid fuse during request-time)
        engine.model.fuse()
    except Exception:
        pass
    try:
        # warm up backend/predictor
        engine.model.predict(dummy, verbose=False)
    except Exception:
        pass

# ----------------------------
# Helpers
# ----------------------------
def encode_jpg(frame_bgr, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()

# ----------------------------
# Endpoints
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Upload MP4. Saves to runs/inputs/<video_id>.mp4 and initializes the job in engine.
    """
    video_id = str(uuid.uuid4())[:12]
    save_path = os.path.join(RUNS_INPUT, f"{video_id}.mp4")

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    meta = engine.init_job(video_id, save_path)
    return {"video_id": video_id, "meta": meta}

@app.get("/test")
async def test():
    return {"video_id": "successful"}




# @app.get("/stream")
# def stream(
#     video_id: str = Query(...),
#     start_frame: int = Query(0, ge=0),
#     downscale: int = Query(640, ge=0),
#     infer_every: int = Query(3, ge=1),
#     blur_ksize: int = Query(25, ge=3),
#     feather_px: int = Query(5, ge=1),
#     outline: bool = Query(False),
#     jpg_quality: int = Query(70, ge=10, le=100),
#     fps: int = Query(10, ge=1, le=60),
# ):
#     boundary = "frame"
#     frame_delay = 1.0 / float(fps)

#     meta = engine.get_meta(video_id)
#     frame_count = int(meta.get("frame_count", -1))

#     def gen():
#         frame_index = start_frame
#         while True:
#             if frame_count > 0 and frame_index >= frame_count:
#                 frame_index = 0

#             with FRAME_LOCK:
#                 res = engine.process_frame_fast(
#                     video_id=video_id,
#                     frame_index=frame_index,
#                     infer_every=infer_every,
#                     downscale=downscale,
#                     blur_ksize=blur_ksize,
#                     feather_px=feather_px,
#                     outline=outline,
#                     outline_strength=0.6,
#                 )

#             jpg = encode_jpg(res["image_bgr"], quality=jpg_quality)

#             yield (
#                 f"--{boundary}\r\n"
#                 "Content-Type: image/jpeg\r\n"
#                 f"Content-Length: {len(jpg)}\r\n\r\n"
#             ).encode("utf-8") + jpg + b"\r\n"

#             frame_index += 1
#             time.sleep(frame_delay)


#     return StreamingResponse(
#         gen(),
#         media_type=f"multipart/x-mixed-replace; boundary={boundary}",
#         headers={
#             "Cache-Control": "no-store",
#             "Pragma": "no-cache",
#             "Connection": "keep-alive",
#             "X-Accel-Buffering": "no",
#         },
#     )
@app.websocket("/ws/stream")
async def ws_stream(
    websocket: WebSocket,
):
    await websocket.accept()

    try:
        # Receive first message as JSON config
        cfg = await websocket.receive_json()
        video_id = cfg["video_id"]

        downscale = int(cfg.get("downscale", 480))
        infer_every = int(cfg.get("infer_every", 5))
        jpg_quality = int(cfg.get("jpg_quality", 70))
        fps = float(cfg.get("fps", 10))
        blur_ksize = int(cfg.get("blur_ksize", 25))
        feather_px = int(cfg.get("feather_px", 5))
        outline = bool(cfg.get("outline", False))

        # Start from 0 always (simple + stable)
        frame_index = 0
        delay = 1.0 / max(1.0, fps)

        # Send "ready"
        await websocket.send_json({"type": "ready"})

        while True:
            # If you want: allow client to ask reset to frame 0 etc (optional)
            # We'll keep it simple.

            with FRAME_LOCK:
                res = engine.process_frame_fast(
                    video_id=video_id,
                    frame_index=frame_index,
                    infer_every=infer_every,
                    downscale=downscale,
                    blur_ksize=blur_ksize,
                    feather_px=feather_px,
                    outline=outline,
                    outline_strength=0.6,
                )

            img = res["image_bgr"]
            payload = encode_jpg(img, quality=jpg_quality)

            # send JSON header then JPEG binary (client will pair them)
            await websocket.send_json({
                "type": "frame",
                "frame_index": frame_index,
                "w": int(img.shape[1]),
                "h": int(img.shape[0]),
                "locked": bool(res.get("locked", False)),
            })
            await websocket.send_bytes(payload)

            frame_index += 1
            await asyncio.sleep(delay)

    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass

@app.post("/select")
def select(
    video_id: str = Form(...),
    frame_index: int = Form(...),
    x: int = Form(...),
    y: int = Form(...),
    downscale: int = Form(640),
):
    with FRAME_LOCK:
        out = engine.select_target(video_id, int(frame_index), int(x), int(y), downscale=downscale)
    return JSONResponse(out)

@app.post("/reset")
def reset(video_id: str = Form(...)):
    return JSONResponse(engine.reset_target(video_id))

@app.post("/render")
def render(
    video_id: str = Form(...),
    downscale: Optional[int] = Form(None),
    blur_ksize: int = Form(31),
    feather_px: int = Form(7),
    outline: bool = Form(False),
    start_frame: int = Form(0),
    end_frame: int = Form(-1),
):
    out_path = os.path.join(RUNS_OUTPUT, f"{video_id}_focused.mp4")

    with FRAME_LOCK:
        result = engine.render_video(
            video_id=video_id,
            out_path=out_path,
            start_frame=int(start_frame),
            end_frame=None if int(end_frame) < 0 else int(end_frame),
            downscale=downscale,
            blur_ksize=int(blur_ksize),
            feather_px=int(feather_px),
            outline=bool(outline),
            fourcc="mp4v",
        )

    return JSONResponse(result)

@app.get("/download")
def download(video_id: str = Query(...)):
    path = os.path.join(RUNS_OUTPUT, f"{video_id}_focused.mp4")
    if not os.path.exists(path):
        return JSONResponse({"ok": False, "error": "file_not_found"}, status_code=404)
    return FileResponse(path, media_type="video/mp4", filename=f"{video_id}_focused.mp4")

@app.post("/close")
def close(video_id: str = Form(...)):
    engine.close_job(video_id)
    return {"ok": True}