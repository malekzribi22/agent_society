#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import time

class TopicChecker(Node):
    def __init__(self):
        super().__init__('topic_checker')
        
    def check_drone_topics(self):
        topic_list = self.get_topic_names_and_types()
        drone_topics = []
        camera_topics = []
        
        for topic_name, topic_type in topic_list:
            if 'drone' in topic_name:
                drone_topics.append((topic_name, topic_type))
            if 'camera' in topic_name:
                camera_topics.append((topic_name, topic_type))
                
        print("=== ALL DRONE TOPICS ===")
        for topic, msg_type in sorted(drone_topics):
            print(f"{topic}")
            
        print("\n=== CAMERA TOPICS ===")
        for topic, msg_type in sorted(camera_topics):
            print(f"{topic}")
            
        return drone_topics

def main():
    rclpy.init()
    node = TopicChecker()
    
    print("Checking for drone and camera topics...")
    
    for i in range(3):
        print(f"\n--- Check {i+1}/3 ---")
        topics = node.check_drone_topics()
        if not topics:
            print("No drone topics found yet. Waiting...")
            time.sleep(2)
        else:
            break
            
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
