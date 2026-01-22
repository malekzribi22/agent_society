#!/usr/bin/env python3
# ROS 2 + YOLOv8 detector with simple color tagging and JSONL logging
# Topics:
#  - Sub:  /drone00/camera/color/image_raw   (sensor_msgs/msg/Image)
#  - Pub:  /drone00/camera/detections_image  (sensor_msgs/msg/Image)
#  - Pub:  /drone00/detections_json          (std_msgs/msg/String JSON)
# File:
#  - /tmp/detections.jsonl   (one JSON per line)

import os, json, time, math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import numpy as np
import cv2

# Ultralytics YOLO
from ultralytics import YOLO
from cv_bridge import CvBridge

CAM_TOPIC = os.environ.get("CAM_TOPIC", "/drone00/camera/color/image_raw")
ANNOT_TOPIC = "/drone00/camera/detections_image"
JSON_TOPIC = "/drone00/detections_json"
JSONL_PATH = "/tmp/detections.jsonl"

# Very simple color naming based on HSV hue
def name_color_bgr_roi(bgr_roi):
    if bgr_roi.size == 0:
        return None
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    h = hsv[...,0].astype(np.float32)
    s = hsv[...,1].astype(np.float32)
    v = hsv[...,2].astype(np.float32)
    # focus on colorful/bright pixels
    mask = (s > 40) & (v > 40)
    if mask.sum() < 50:
        return "unknown"
    h_sel = h[mask]
    # average hue
    mean_h = float(np.mean(h_sel))
    # hue ranges (OpenCV: H in [0..179])
    if mean_h < 10 or mean_h >= 170: return "red"
    if 10 <= mean_h < 25:  return "orange"
    if 25 <= mean_h < 35:  return "yellow"
    if 35 <= mean_h < 85:  return "green"
    if 85 <= mean_h < 115: return "cyan"
    if 115 <= mean_h < 140:return "blue"
    if 140 <= mean_h < 170:return "magenta"
    return "unknown"

class YoloDetector(Node):
    def __init__(self):
        super().__init__("yolo_detector")
        self.bridge = CvBridge()
        self.get_logger().info(f"Subscribing to: {CAM_TOPIC}")
        self.get_logger().info(f"Annotated image pub: {ANNOT_TOPIC}")
        self.get_logger().info(f"Detections JSON pub: {JSON_TOPIC}")
        self.get_logger().info(f"JSONL path (optional): {JSONL_PATH}")

        self.sub = self.create_subscription(Image, CAM_TOPIC, self.cb, 10)
        self.pub_img = self.create_publisher(Image, ANNOT_TOPIC, 10)
        self.pub_json = self.create_publisher(String, JSON_TOPIC, 10)

        # Load a tiny model to stay fast
        self.model = YOLO(os.environ.get("YOLO_MODEL","yolov8n.pt"))
        self.get_logger().info("YOLOv8n loaded.")

        # open JSONL once
        self.jsonl = open(JSONL_PATH, "a", buffering=1)

    def cb(self, msg: Image):
        try:
            cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge: {e}")
            return

        h, w = cv.shape[:2]
        res = self.model.predict(cv, imgsz=min(w,h), conf=0.45, verbose=False)[0]

        dets = []
        # draw
        for b in res.boxes:
            cls_id = int(b.cls[0])
            cls_name = self.model.names.get(cls_id, str(cls_id))
            conf = float(b.conf[0])
            x1,y1,x2,y2 = [int(v) for v in b.xyxy[0].tolist()]

            # simple color tag for people only (or for all classes)
            color_tag = None
            roi = cv[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
            if roi.size > 0:
                color_tag = name_color_bgr_roi(roi)

            det = {
                "ts": time.time(),
                "class": cls_name,
                "conf": conf,
                "bbox_xyxy": [x1,y1,x2,y2],
                "image_size": [w,h],
                "dominant_color": color_tag
            }
            dets.append(det)

            # annotate
            label = f"{cls_name} {conf:.2f}"
            if color_tag: label += f" ({color_tag})"
            cv2.rectangle(cv, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(cv, label, (x1, max(0,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        # publish image
        out = self.bridge.cv2_to_imgmsg(cv, encoding="bgr8")
        out.header = msg.header
        self.pub_img.publish(out)

        # publish JSON
        payload = {"ts": time.time(), "detections": dets}
        s = String()
        s.data = json.dumps(payload)
        self.pub_json.publish(s)

        # append JSONL
        try:
            self.jsonl.write(json.dumps(payload) + "\n")
        except Exception:
            pass

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

