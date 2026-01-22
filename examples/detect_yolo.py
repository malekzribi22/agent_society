#!/usr/bin/env python3
import os, json, time
from datetime import datetime
import numpy as np, cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import String

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError(
        "Ultralytics not found. Install with:\n"
        "  python3 -m pip install --user ultralytics opencv-python\n"
    ) from e

def sensor_sub_qos():
    # Camera topics usually BEST_EFFORT (fast & lossy)
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
    )

def reliable_pub_qos():
    # Viewers like image_view default to RELIABLE; match that
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )

class YoloDetector(Node):
    def __init__(self):
        super().__init__("yolo_detector")

        self.cam_topic = os.environ.get("CAM_TOPIC", "/drone00/camera/color/image_raw")
        self.ann_topic = os.environ.get("ANN_TOPIC", "/drone00/camera/detections_image")
        self.json_topic = os.environ.get("DET_JSON_TOPIC", "/drone00/detections_json")
        self.jsonl_path = os.environ.get("DET_JSONL", "/tmp/detections.jsonl")

        model_name = os.environ.get("YOLO_MODEL", "yolov8n.pt")
        self.model = YOLO(model_name)
        self.get_logger().info("YOLOv8n loaded.")

        self.imgsz = int(os.environ.get("YOLO_IMGSZ", "1216"))  # multiple of 32 to avoid warnings

        # QoS: sub BEST_EFFORT, pubs RELIABLE
        sub_qos = sensor_sub_qos()
        pub_qos = reliable_pub_qos()

        self.pub_ann = self.create_publisher(Image, self.ann_topic, pub_qos)
        self.pub_json = self.create_publisher(String, self.json_topic, pub_qos)

        self.get_logger().info(f"Subscribing to: {self.cam_topic}")
        self.get_logger().info(f"Annotated pub:  {self.ann_topic}")
        self.get_logger().info(f"JSON pub:       {self.json_topic}")
        self.get_logger().info(f"JSONL file:     {self.jsonl_path}")

        self.sub = self.create_subscription(Image, self.cam_topic, self.cb, sub_qos)

        self._last_log = time.time()
        self._frames = 0

    def cb(self, msg: Image):
        try:
            h, w = msg.height, msg.width
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            if arr.size != h * w * 3:
                self.get_logger().warn(
                    f"Image buffer size mismatch (got {arr.size}, expected {h*w*3}); skipping.")
                return
            frame = arr.reshape((h, w, 3))
            if msg.encoding.lower() == "rgb8":
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                bgr = frame

            results = self.model.predict(bgr, imgsz=self.imgsz, conf=0.45, verbose=False)
            res = results[0]
            ann_bgr = res.plot()

            out = Image()
            out.header = msg.header
            out.height = h
            out.width = w
            out.encoding = "bgr8"
            out.is_bigendian = 0
            out.step = w * 3
            out.data = ann_bgr.tobytes()
            self.pub_ann.publish(out)

            dets = []
            names = res.names
            if res.boxes is not None and len(res.boxes) > 0:
                boxes = res.boxes
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    xyxy = boxes.xyxy[i].tolist()
                    dets.append({
                        "class_id": cls_id,
                        "class_name": names.get(cls_id, str(cls_id)),
                        "confidence": round(conf, 4),
                        "xyxy": [round(float(v), 2) for v in xyxy],
                    })

            stamp = {"sec": int(msg.header.stamp.sec), "nanosec": int(msg.header.stamp.nanosec)}
            payload = {
                "timestamp": stamp,
                "frame_id": msg.header.frame_id or "camera",
                "width": w,
                "height": h,
                "detections": dets,
                "t": datetime.utcnow().isoformat() + "Z",
            }
            s = String()
            s.data = json.dumps(payload)
            self.pub_json.publish(s)

            try:
                with open(self.jsonl_path, "a", encoding="utf-8") as f:
                    f.write(s.data + "\n")
            except Exception:
                pass

            self._frames += 1
            now = time.time()
            if now - self._last_log > 2.0:
                fps = self._frames / (now - self._last_log)
                self.get_logger().info(f"processing ~{fps:.1f} fps")
                self._last_log = now
                self._frames = 0

        except Exception as e:
            self.get_logger().error(f"detector callback error: {e}")

def main():
    rclpy.init()
    node = YoloDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

