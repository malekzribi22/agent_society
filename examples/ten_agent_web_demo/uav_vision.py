"""YOLO helpers for UAV car counting."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:  # pragma: no cover - optional dependency
    from ultralytics import YOLO
except Exception:  # pragma: no cover - tolerate environments w/o ultralytics
    YOLO = None  # type: ignore[assignment]


class UavYoloCarCounter:
    def __init__(self, weights_path: str = "yolov8n.pt"):
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO is not installed")
        self.model = YOLO(weights_path)
        # COCO vehicle classes: car (2), bus (5), truck (7)
        self.vehicle_ids = {2, 5, 7}

    def count_cars(self, image_path: str) -> int:
        path = Path(image_path)
        if not path.exists():
            return 0

        img = Image.open(path).convert("RGB")
        w, h = img.size
        # Resize small images to improve detection of small objects
        if max(w, h) < 1024:
            scale = 1024 / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)))

        results = self.model.predict(
            np.array(img),
            conf=0.05,
            iou=0.45,
            imgsz=1920,
            verbose=False,
        )
        if not results:
            return 0

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return 0

        count = 0
        for box in boxes:
            try:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
            except Exception:
                continue
            
            # Filter by class, confidence, and area
            if cls_id in self.vehicle_ids and conf >= 0.05:
                # Calculate area to filter tiny noisy boxes
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > 200:
                    count += 1

        return count


_UAV_DETECTOR: Optional[UavYoloCarCounter] = None


def get_uav_detector() -> Optional[UavYoloCarCounter]:
    """Return a shared YOLO detector instance if available."""

    global _UAV_DETECTOR
    if _UAV_DETECTOR is None:
        try:
            _UAV_DETECTOR = UavYoloCarCounter()
        except Exception as exc:  # pragma: no cover - fall back to dummy mode
            logging.warning("[ten_agent_web_demo] Unable to initialize YOLO detector: %s", exc)
            _UAV_DETECTOR = None
    return _UAV_DETECTOR
