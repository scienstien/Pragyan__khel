import os
import cv2
from src.focus_engine import FocusEngine, encode_jpg_b64

VID = "demo1"
VIDEO_PATH = "runs/inputs/demo.mp4"

engine = FocusEngine(model_path="models/yolov8n-seg.pt", device="0")
meta = engine.init_job(VID, VIDEO_PATH)
print(meta)

res = engine.process_frame(VID, 0, downscale=720)
jpg = encode_jpg_b64(res["image_bgr"])
print("preview bytes", len(jpg))

h, w = res["image_bgr"].shape[:2]
engine.select_target(VID, 0, w // 2, h // 2)

for i in range(1, 30):
    r = engine.process_frame(VID, i, downscale=720)
    cv2.imwrite(f"runs/jobs/{VID}/preview_{i:03d}.jpg", r["image_bgr"])

out = engine.render_video(VID, f"runs/outputs/{VID}_focused.mp4", start_frame=0, end_frame=150, downscale=720)
print(out)

engine.close_job(VID)