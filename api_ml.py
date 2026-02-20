import os
import uuid
import shutil
from typing import Optional

import cv2
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse

from src.focus_engine import FocusEngine

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

# Allow frontend (React) to call this during hackathon
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later if needed
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


@app.get("/frame")
def get_frame(
    video_id: str = Query(...),
    frame_index: int = Query(..., ge=0),

    # Preview parameters
    downscale: int = Query(640, ge=0),
    infer_every: int = Query(3, ge=1),
    blur_ksize: int = Query(25, ge=3),
    feather_px: int = Query(5, ge=1),
    outline: bool = Query(False),
    jpg_quality: int = Query(85, ge=10, le=100),
):
    """
    Returns a preview frame as image/jpeg.
    Uses process_frame_fast for speed (inference every N frames).
    """
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

    # Optional: you can also attach state via headers if your frontend wants it
    # (kept simple for now)

    return StreamingResponse(iter([payload]), media_type="image/jpeg")


@app.post("/select")
def select(
    video_id: str = Form(...),
    frame_index: int = Form(...),
    x: int = Form(...),
    y: int = Form(...),

    # CRITICAL: must match /frame downscale for correct click behavior
    downscale: int = Form(640),
):
    out = engine.select_target(video_id, int(frame_index), int(x), int(y), downscale=downscale)
    return JSONResponse(out)


@app.post("/reset")
def reset(video_id: str = Form(...)):
    return JSONResponse(engine.reset_target(video_id))


@app.post("/render")
def render(
    video_id: str = Form(...),

    # If you want quick demo render: set downscale=720 or 640
    # For final quality: downscale=None
    downscale: Optional[int] = Form(None),

    blur_ksize: int = Form(31),
    feather_px: int = Form(7),
    outline: bool = Form(False),

    start_frame: int = Form(0),
    end_frame: int = Form(-1),
):
    out_path = os.path.join(RUNS_OUTPUT, f"{video_id}_focused.mp4")

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
    """
    Optional: call when frontend is done with a video to free RAM/capture.
    """
    engine.close_job(video_id)
    return {"ok": True}