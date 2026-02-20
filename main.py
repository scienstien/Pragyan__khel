import os
import cv2
import uuid

from src.focus_engine import FocusEngine


WINDOW = "SmartFocus Preview"

clicked = {"do": False, "x": 0, "y": 0}
paused = {"on": False}


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked["do"] = True
        clicked["x"] = x
        clicked["y"] = y


def map_display_to_frame(xd, yd, disp_w, disp_h, frame_w, frame_h):
    sx = frame_w / float(disp_w)
    sy = frame_h / float(disp_h)
    return int(xd * sx), int(yd * sy)


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

    fps = meta["fps"] if meta["fps"] and 1 <= meta["fps"] <= 120 else 30.0
    total_frames = meta["frame_count"] if meta["frame_count"] and meta["frame_count"] > 0 else 10**9

    frame_w = meta["width"]
    frame_h = meta["height"]

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
            if not paused["on"]:
                res = engine.process_frame(
                    video_id=video_id,
                    frame_index=frame_idx,
                    downscale=None,
                    blur_ksize=31,
                    feather_px=7,
                    outline=True,
                    outline_strength=0.6,
                )
            else:
                res = engine.process_frame(
                    video_id=video_id,
                    frame_index=frame_idx,
                    downscale=None,
                    blur_ksize=31,
                    feather_px=7,
                    outline=True,
                    outline_strength=0.6,
                )

            img = res["image_bgr"]
            out = cv2.resize(img, (display_w, display_h), interpolation=cv2.INTER_AREA)

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

            if clicked["do"]:
                xf, yf = map_display_to_frame(
                    clicked["x"], clicked["y"],
                    display_w, display_h,
                    img.shape[1], img.shape[0],
                )
                sel = engine.select_target(video_id, frame_idx, xf, yf)
                print("select:", sel)
                clicked["do"] = False

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
                frame_idx += 1

    finally:
        engine.close_job(video_id)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()