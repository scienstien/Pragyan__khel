import os
import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


BBox = Tuple[int, int, int, int]  # x1,y1,x2,y2


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _clamp_bbox(b: BBox, w: int, h: int) -> BBox:
    x1, y1, x2, y2 = b
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def _iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _gaussian_blur(frame: np.ndarray, k: int = 31) -> np.ndarray:
    k = int(k)
    if k % 2 == 0:
        k += 1
    k = max(3, k)
    return cv2.GaussianBlur(frame, (k, k), 0)


def _mask_feather(mask: np.ndarray, feather_px: int = 7) -> np.ndarray:
    feather_px = max(1, int(feather_px))
    k = feather_px if feather_px % 2 == 1 else feather_px + 1
    m = cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)
    return np.clip(m, 0.0, 1.0)


def _mask_to_outline(mask01: np.ndarray) -> np.ndarray:
    m = (mask01 > 0.5).astype(np.uint8) * 255
    edges = cv2.Canny(m, 50, 150)
    return edges


@dataclass
class TargetState:
    locked: bool = False
    target_track_id: Optional[int] = None
    last_bbox: Optional[BBox] = None
    miss_count: int = 0
    max_misses: int = 15
    last_mask_ema: Optional[np.ndarray] = None
    ema_alpha: float = 0.7

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.last_mask_ema is not None:
            d["last_mask_ema"] = None
        return d


class FocusEngine:
    """
    Backend-callable engine:
    - load YOLOv8-seg
    - run tracking (ByteTrack/BoT-SORT) per frame via ultralytics .track()
    - click-to-select by point-in-mask (fallback bbox)
    - keep target stable by track ID; ignore class changes
    - blur-composite using instance mask
    """

    def __init__(
        self,
        model_path: str = "yolov8n-seg.pt",
        device: str = "0",
        tracker: str = "bytetrack.yaml",
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 640,
        job_root: str = "runs/jobs",
        
    ):  
        self._cap_pos: Dict[str, int] = {}
        self.model_path = model_path
        self.device = device
        self.tracker = tracker
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.job_root = job_root
        _ensure_dir(self.job_root)

        self.model = YOLO(self.model_path)
        self._video_caps: Dict[str, cv2.VideoCapture] = {}
        self._states: Dict[str, TargetState] = {}

    def init_job(self, video_id: str, video_path: str) -> Dict[str, Any]:
        job_dir = os.path.join(self.job_root, video_id)
        _ensure_dir(job_dir)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else -1

        meta = {
            "video_id": video_id,
            "video_path": os.path.abspath(video_path),
            "fps": fps,
            "width": w,
            "height": h,
            "frame_count": n,
            "created_at": time.time(),
        }
        with open(os.path.join(job_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        self._video_caps[video_id] = cap
        self._cap_pos.pop(video_id, None)
        self._states[video_id] = TargetState()

        return meta

    def close_job(self, video_id: str) -> None:
        cap = self._video_caps.get(video_id)
        if cap is not None:
            cap.release()
        self._video_caps.pop(video_id, None)
        self._cap_pos.pop(video_id, None)
        self._states.pop(video_id, None)

    def get_meta(self, video_id: str) -> Dict[str, Any]:
        job_dir = os.path.join(self.job_root, video_id)
        with open(os.path.join(job_dir, "meta.json"), "r", encoding="utf-8") as f:
            return json.load(f)

    def _read_frame(self, video_id: str, frame_index: int) -> np.ndarray:
        cap = self._video_caps.get(video_id)
        if cap is None:
            raise RuntimeError(f"Job not initialized: {video_id}")

        frame_index = int(frame_index)
        cur = int(self._cap_pos.get(video_id, 0))

        # If we're asking for the next frame (or same), avoid expensive seek
        if frame_index == cur:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Cannot read frame {frame_index} for {video_id}")
            self._cap_pos[video_id] = cur + 1
            return frame

        # If small forward jump, read-and-discard instead of cap.set()
        if frame_index > cur and (frame_index - cur) <= 10:
            frame = None
            for _ in range(frame_index - cur + 1):
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Cannot read frame {frame_index} for {video_id}")
            self._cap_pos[video_id] = frame_index + 1
            return frame

        # Otherwise do a seek (slow but acceptable for scrubbing)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Cannot read frame {frame_index} for {video_id}")
        self._cap_pos[video_id] = frame_index + 1
        return frame

    def _infer_track(self, frame: np.ndarray) -> Any:
        res = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        return res[0]

    def _extract_instances(self, res0: Any, frame_shape: Tuple[int, int, int]) -> List[Dict[str, Any]]:
        H, W = frame_shape[:2]
        instances: List[Dict[str, Any]] = []

        boxes = res0.boxes
        if boxes is None or len(boxes) == 0:
            return instances

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros((len(xyxy),), dtype=int)
        conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones((len(xyxy),), dtype=float)

        track_ids = None
        if getattr(boxes, "id", None) is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)

        masks = None
        if getattr(res0, "masks", None) is not None and res0.masks is not None:
            masks = res0.masks.data.cpu().numpy()

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(int, xyxy[i])
            x1, y1, x2, y2 = _clamp_bbox((x1, y1, x2, y2), W, H)

            tid = int(track_ids[i]) if track_ids is not None else None

            inst = {
                "track_id": tid,
                "bbox": (x1, y1, x2, y2),
                "cls": int(cls[i]),
                "conf": float(conf[i]),
                "mask": None,
            }

            if masks is not None and i < masks.shape[0]:
                m = masks[i]
                if m.shape[-2:] != (H, W):
                    m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
                inst["mask"] = m.astype(np.float32)

            instances.append(inst)

        return instances

    def select_target(self, video_id: str, frame_index: int, x: int, y: int, downscale: Optional[int] = None) -> Dict[str, Any]:
        frame = self._read_frame(video_id, frame_index)

        # apply the SAME downscale used in preview so click coords match
        if downscale is not None and downscale > 0:
            h, w = frame.shape[:2]
            scale = float(downscale) / max(h, w)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        res0 = self._infer_track(frame)
        instances = self._extract_instances(res0, frame.shape)
        instances = self._extract_instances(res0, frame.shape)

        pick = None
        best_area = None

        for inst in instances:
            x1, y1, x2, y2 = inst["bbox"]
            if not (x1 <= x <= x2 and y1 <= y <= y2):
                continue

            m = inst["mask"]
            if m is not None:
                if y < 0 or y >= m.shape[0] or x < 0 or x >= m.shape[1]:
                    continue
                if m[y, x] < 0.5:
                    continue

            area = (x2 - x1) * (y2 - y1)
            if best_area is None or area < best_area:
                best_area = area
                pick = inst

        if pick is None:
            return {"selected": False, "reason": "no_instance_at_click"}

        st = self._states[video_id]
        st.locked = True
        st.target_track_id = pick["track_id"]
        st.last_bbox = pick["bbox"]
        st.miss_count = 0
        st.last_mask_ema = None

        self._persist_state(video_id)

        return {
            "selected": True,
            "target_track_id": st.target_track_id,
            "bbox": list(st.last_bbox),
            "cls": pick["cls"],
            "conf": pick["conf"],
        }

    def reset_target(self, video_id: str) -> Dict[str, Any]:
        st = self._states[video_id]
        st.locked = False
        st.target_track_id = None
        st.last_bbox = None
        st.miss_count = 0
        st.last_mask_ema = None
        self._persist_state(video_id)
        return {"ok": True}

    def set_max_misses(self, video_id: str, n: int) -> Dict[str, Any]:
        st = self._states[video_id]
        st.max_misses = int(n)
        self._persist_state(video_id)
        return {"ok": True, "max_misses": st.max_misses}
    def process_frame_fast(
    self,
    video_id: str,
    frame_index: int,
    infer_every: int = 3,
    blur_ksize: int = 25,
    feather_px: int = 5,
    outline: bool = False,
    outline_strength: float = 0.6,
    downscale: Optional[int] = 640,
) -> Dict[str, Any]:
        """
        Fast preview:
        - Full YOLO+tracking inference only every `infer_every` frames.
        - In between, reuse cached mask EMA and just composite blur (cheap).
        """
        st = self._states[video_id]
        frame = self._read_frame(video_id, frame_index)

        # consistent downscale for both inference and non-inference frames
        if downscale is not None and downscale > 0:
            h, w = frame.shape[:2]
            scale = float(downscale) / max(h, w)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        H, W = frame.shape[:2]

        do_infer = (frame_index % max(1, int(infer_every)) == 0) or (st.last_mask_ema is None)

        selected_inst = None

        if do_infer:
            res0 = self._infer_track(frame)
            instances = self._extract_instances(res0, frame.shape)

            # 1) try lock by track ID
            if st.locked and st.target_track_id is not None:
                for inst in instances:
                    if inst["track_id"] is not None and inst["track_id"] == st.target_track_id:
                        selected_inst = inst
                        break

            # 2) fallback by IOU with last bbox
            if selected_inst is None and st.locked and st.last_bbox is not None:
                best = None
                best_score = 0.0
                for inst in instances:
                    score = _iou(st.last_bbox, inst["bbox"])
                    if score > best_score:
                        best_score = score
                        best = inst
                if best is not None and best_score > 0.2:
                    selected_inst = best
                    if best["track_id"] is not None:
                        st.target_track_id = best["track_id"]

            # handle lost
            if st.locked and selected_inst is None:
                st.miss_count += 1
                dropped = st.miss_count >= st.max_misses
                if dropped:
                    st.locked = False
                    st.target_track_id = None
                    st.last_bbox = None
                    st.last_mask_ema = None
                    st.miss_count = 0
                self._persist_state(video_id)
                return {
                    "locked": False if dropped else True,
                    "frame_index": frame_index,
                    "miss_count": st.miss_count,
                    "dropped": dropped,
                    "image_bgr": frame,
                }

            # update mask cache if we have a selected instance
            if st.locked and selected_inst is not None:
                st.miss_count = 0
                st.last_bbox = selected_inst["bbox"]

                m = selected_inst["mask"]
                if m is None:
                    x1, y1, x2, y2 = selected_inst["bbox"]
                    m = np.zeros((H, W), dtype=np.float32)
                    m[y1:y2, x1:x2] = 1.0

                if st.last_mask_ema is None:
                    st.last_mask_ema = m.copy()
                else:
                    st.last_mask_ema = st.ema_alpha * m + (1.0 - st.ema_alpha) * st.last_mask_ema

                st.last_mask_ema = np.clip(st.last_mask_ema, 0.0, 1.0)

        # If not locked, just show original
        if not st.locked:
            return {
                "locked": False,
                "frame_index": frame_index,
                "miss_count": st.miss_count,
                "dropped": False,
                "image_bgr": frame,
            }

        # If locked but mask not available yet, show original
        if st.last_mask_ema is None:
            return {
                "locked": True,
                "frame_index": frame_index,
                "target_track_id": st.target_track_id,
                "bbox": list(st.last_bbox) if st.last_bbox else None,
                "miss_count": st.miss_count,
                "dropped": False,
                "image_bgr": frame,
            }

        # Composite with cached mask (cheap path)
        mask01 = _mask_feather(st.last_mask_ema, feather_px=feather_px)
        blurred = _gaussian_blur(frame, k=blur_ksize)

        mask3 = np.repeat(mask01[:, :, None], 3, axis=2).astype(np.float32)
        out = (frame.astype(np.float32) * mask3 + blurred.astype(np.float32) * (1.0 - mask3)).astype(np.uint8)

        if outline:
            edges = _mask_to_outline(mask01)
            edge_mask = edges > 0
            out = out.copy()
            factor = 1.0 + float(outline_strength)
            tmp = out[edge_mask].astype(np.float32) * factor
            out[edge_mask] = np.clip(tmp, 0, 255).astype(np.uint8)

        self._persist_state(video_id)

        return {
            "locked": True,
            "frame_index": frame_index,
            "target_track_id": st.target_track_id,
            "bbox": list(st.last_bbox) if st.last_bbox else None,
            "miss_count": st.miss_count,
            "dropped": False,
            "image_bgr": out,
        }

    def process_frame(
        self,
        video_id: str,
        frame_index: int,
        blur_ksize: int = 31,
        feather_px: int = 7,
        outline: bool = True,
        outline_strength: float = 0.6,
        downscale: Optional[int] = None,
    ) -> Dict[str, Any]:
        st = self._states[video_id]
        frame = self._read_frame(video_id, frame_index)

        if downscale is not None and downscale > 0:
            h, w = frame.shape[:2]
            scale = float(downscale) / max(h, w)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        H, W = frame.shape[:2]
        res0 = self._infer_track(frame)
        instances = self._extract_instances(res0, frame.shape)

        selected_inst = None

        if st.locked and st.target_track_id is not None:
            for inst in instances:
                if inst["track_id"] is not None and inst["track_id"] == st.target_track_id:
                    selected_inst = inst
                    break

        if selected_inst is None and st.locked and st.last_bbox is not None:
            best = None
            best_score = 0.0
            for inst in instances:
                score = _iou(st.last_bbox, inst["bbox"])
                if score > best_score:
                    best_score = score
                    best = inst
            if best is not None and best_score > 0.2:
                selected_inst = best
                if best["track_id"] is not None:
                    st.target_track_id = best["track_id"]

        if not st.locked:
            return {
                "locked": False,
                "frame_index": frame_index,
                "miss_count": st.miss_count,
                "dropped": False,
                "image_bgr": frame,
            }

        if selected_inst is None:
            st.miss_count += 1
            dropped = st.miss_count >= st.max_misses
            if dropped:
                st.locked = False
                st.target_track_id = None
                st.last_bbox = None
                st.last_mask_ema = None
                st.miss_count = 0
            self._persist_state(video_id)
            return {
                "locked": False if dropped else True,
                "frame_index": frame_index,
                "miss_count": st.miss_count,
                "dropped": dropped,
                "image_bgr": frame,
            }

        st.miss_count = 0
        st.last_bbox = selected_inst["bbox"]

        m = selected_inst["mask"]
        if m is None:
            x1, y1, x2, y2 = selected_inst["bbox"]
            m = np.zeros((H, W), dtype=np.float32)
            m[y1:y2, x1:x2] = 1.0

        if st.last_mask_ema is None:
            st.last_mask_ema = m.copy()
        else:
            st.last_mask_ema = st.ema_alpha * m + (1.0 - st.ema_alpha) * st.last_mask_ema

        mask01 = np.clip(st.last_mask_ema, 0.0, 1.0)
        mask01 = _mask_feather(mask01, feather_px=feather_px)

        blurred = _gaussian_blur(frame, k=blur_ksize)

        mask3 = np.repeat(mask01[:, :, None], 3, axis=2).astype(np.float32)
        out = (frame.astype(np.float32) * mask3 + blurred.astype(np.float32) * (1.0 - mask3)).astype(np.uint8)
        if outline:
            edges = _mask_to_outline(mask01)
            edge_mask = edges > 0  # 2D boolean

            out = out.copy()
            factor = 1.0 + float(outline_strength)
            tmp = out[edge_mask].astype(np.float32) * factor
            out[edge_mask] = np.clip(tmp, 0, 255).astype(np.uint8)
        self._persist_state(video_id)

        return {
            "locked": True,
            "frame_index": frame_index,
            "target_track_id": st.target_track_id,
            "bbox": list(st.last_bbox),
            "miss_count": st.miss_count,
            "dropped": False,
            "image_bgr": out,
        }

    def render_video(
        self,
        video_id: str,
        out_path: str,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        blur_ksize: int = 31,
        feather_px: int = 7,
        outline: bool = False,
        downscale: Optional[int] = None,
        fourcc: str = "mp4v",
    ) -> Dict[str, Any]:
        meta = self.get_meta(video_id)
        fps = meta["fps"]

        cap = self._video_caps.get(video_id)
        if cap is None:
            raise RuntimeError(f"Job not initialized: {video_id}")

        total = int(meta["frame_count"]) if meta.get("frame_count", -1) != -1 else None
        if end_frame is None:
            end_frame = total - 1 if total is not None else start_frame + 300

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
        ok, frame0 = cap.read()
        if not ok or frame0 is None:
            raise RuntimeError("Cannot read start frame")

        if downscale is not None and downscale > 0:
            h, w = frame0.shape[:2]
            scale = float(downscale) / max(h, w)
            if scale < 1.0:
                frame0 = cv2.resize(frame0, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        H, W = frame0.shape[:2]
        _ensure_dir(os.path.dirname(out_path) or ".")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc), fps, (W, H))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open writer: {out_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

        processed = 0
        for fi in range(int(start_frame), int(end_frame) + 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            res = self.process_frame(
                video_id=video_id,
                frame_index=fi,
                blur_ksize=blur_ksize,
                feather_px=feather_px,
                outline=outline,
                downscale=downscale,
            )
            out = res["image_bgr"]
            writer.write(out)
            processed += 1

        writer.release()

        return {
            "ok": True,
            "out_path": os.path.abspath(out_path),
            "processed_frames": processed,
            "fps": fps,
        }

    def _persist_state(self, video_id: str) -> None:
        job_dir = os.path.join(self.job_root, video_id)
        st = self._states[video_id]
        p = os.path.join(job_dir, "state.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(st.to_json(), f, indent=2)


def encode_jpg_b64(frame_bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()