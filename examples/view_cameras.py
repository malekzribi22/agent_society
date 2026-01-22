#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class DualCameraViewer(Node):
    def __init__(self):
        super().__init__('dual_camera_viewer')
        self.bridge = CvBridge()
        
        # Subscribe to both cameras
        self.color_sub = self.create_subscription(
            Image, '/drone/camera_color/color/image_raw', 
            self.color_callback, 10)
        
        self.depth_sub = self.create_subscription(
            Image, '/drone/camera_depth/depth/image_raw',
            self.depth_callback, 10)
            
        self.get_logger().info("Dual camera viewer started. Waiting for images...")
        self.depth_received = False
        
    def color_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imshow('RGB Camera', cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Color error: {e}")
            
    def depth_callback(self, msg):
        try:
            self.depth_received = True
            # Convert depth image (32FC1 format)
            depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            
            # Normalize for visualization
            depth_visual = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_colormap = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)
            
            cv2.imshow('Depth Camera', depth_colormap)
            cv2.waitKey(1)
            
            # Log depth statistics occasionally
            if np.random.random() < 0.05:  # 5% chance
                valid_depths = depth_image[depth_image > 0]
                if len(valid_depths) > 0:
                    self.get_logger().info(f"Depth range: {valid_depths.min():.2f} - {valid_depths.max():.2f} m")
                    
        except Exception as e:
            if not self.depth_received:
                self.get_logger().warn("Depth data not available yet or wrong format")
            self.get_logger().error(f"Depth processing error: {e}")

def main():
    rclpy.init()
    node = DualCameraViewer()
    
    print("Dual camera viewer started!")
    print("Looking for:")
    print("  - /drone/camera_color/color/image_raw")
    print("  - /drone/camera_depth/depth/image_raw")
    print("Press Ctrl+C to exit")
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
