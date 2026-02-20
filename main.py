import os
import cv2
import uuid
import numpy as np

from src.focus_engine import FocusEngine

WINDOW = "SmartFocus Preview"

# Speed/quality knobs
PREVIEW_DOWNSCALE = 640     # 480/640/720
INFER_EVERY = 3             # 2–4 (higher = faster, less responsive)
FRAME_SKIP = 5             # keep 1 with process_frame_fast
BLUR_KSIZE = 25             # 21/25/31
FEATHER_PX = 5              # 3–7
OUTLINE = False             # True costs a bit

clicked = {"do": False, "x": 0, "y": 0}
paused = {"on": False}


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked["do"] = True
        clicked["x"] = x
        clicked["y"] = y


def letterbox_to_display(img, disp_w, disp_h):
    ih, iw = img.shape[:2]
    scale = min(disp_w / iw, disp_h / ih)
    new_w = int(iw * scale)
    new_h = int(ih * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
    off_x = (disp_w - new_w) // 2
    off_y = (disp_h - new_h) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    return canvas


def map_click_letterbox(xd, yd, disp_w, disp_h, img_w, img_h):
    scale = min(disp_w / img_w, disp_h / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    off_x = (disp_w - new_w) // 2
    off_y = (disp_h - new_h) // 2

    # Click in black bars -> ignore
    if xd < off_x or xd >= off_x + new_w or yd < off_y or yd >= off_y + new_h:
        return None

    xf = int((xd - off_x) / scale)
    yf = int((yd - off_y) / scale)

    xf = max(0, min(img_w - 1, xf))
    yf = max(0, min(img_h - 1, yf))
    return xf, yf


def main():
    video_path = os.path.join("runs", "inputs", "demo.mp4")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found at: {video_path}")

    engine = FocusEngine(
        model_path=os.path.join("models", "yolov8n-seg.pt"),
        device="0",
        tracker="bytetrack.yaml",
        conf=0.25,
        iou=0.5,
        imgsz=640,
        job_root=os.path.join("runs", "jobs"),
    )

    video_id = "local_" + str(uuid.uuid4())[:8]
    meta = engine.init_job(video_id, video_path)

    total_frames = meta["frame_count"] if meta.get("frame_count", 0) and meta["frame_count"] > 0 else 10**9
    frame_w = meta["width"]
    frame_h = meta["height"]

    # Window size (any size is OK, mapping will handle it)
    display_w = 960
    display_h = int(frame_h * (display_w / frame_w))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, display_w, display_h)
    cv2.setMouseCallback(WINDOW, on_mouse)

    frame_idx = 0

    print("Controls:")
    print("  Left click : select subject at click point (locks focus)")
    print("  r          : reset focus")
    print("  space      : pause/resume")
    print("  a / d      : step -1 / +1 frame (when paused)")
    print("  esc        : exit")

    try:
        while frame_idx < total_frames:
            # Fast preview (inference only every INFER_EVERY frames)
            res = engine.process_frame_fast(
                video_id=video_id,
                frame_index=frame_idx,
                infer_every=INFER_EVERY,
                downscale=PREVIEW_DOWNSCALE,
                blur_ksize=BLUR_KSIZE,
                feather_px=FEATHER_PX,
                outline=OUTLINE,
                outline_strength=0.6,
            )

            img = res["image_bgr"]
            out = letterbox_to_display(img, display_w, display_h)

            status = "LOCKED" if res.get("locked", False) else "IDLE"
            if res.get("dropped", False):
                status = "DROPPED"

            cv2.putText(
                out,
                f"frame={frame_idx}  status={status}  target_id={res.get('target_track_id', None)}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if paused["on"]:
                cv2.putText(
                    out,
                    "PAUSED",
                    (12, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(WINDOW, out)

            # Handle click exactly once
            if clicked["do"]:
                mapped = map_click_letterbox(
                    clicked["x"], clicked["y"],
                    display_w, display_h,
                    img.shape[1], img.shape[0],
                )

                if mapped is None:
                    print("click ignored (black bars)")
                else:
                    xf, yf = mapped
                    sel = engine.select_target(video_id, frame_idx, xf, yf, downscale=PREVIEW_DOWNSCALE)
                    print("select:", sel)

                clicked["do"] = False  # reset click

            wait = 1 if not paused["on"] else 0
            key = cv2.waitKey(wait) & 0xFF

            if key == 27:
                break
            elif key == ord(" "):
                paused["on"] = not paused["on"]
            elif key == ord("r"):
                print("reset:", engine.reset_target(video_id))
            elif paused["on"] and key == ord("a"):
                frame_idx = max(0, frame_idx - 1)
                continue
            elif paused["on"] and key == ord("d"):
                frame_idx = min(total_frames - 1, frame_idx + 1)
                continue

            if not paused["on"]:
                frame_idx += FRAME_SKIP

    finally:
        engine.close_job(video_id)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()