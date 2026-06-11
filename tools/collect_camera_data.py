"""
Collect training images from the car's camera for labeling.

Run on the car (ROS2) while driving laps with the other car(s) on track:
    python3 tools/collect_camera_data.py --topic /camera/color/image_raw \
        --out data/car_images --every 5

Saves every Nth frame as a JPEG.  Then label the cars (one class: `car`) with
any tool that exports YOLO format — Roboflow, Label Studio, or labelImg — and
point tools/train_car_detector.py at the result.
"""

import argparse
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic', default='/camera/color/image_raw')
    ap.add_argument('--out', default='data/car_images')
    ap.add_argument('--every', type=int, default=5, help='save every Nth frame')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import cv2
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image

    class Collector(Node):
        def __init__(self):
            super().__init__('camera_data_collector')
            self.n = 0
            self.create_subscription(Image, args.topic, self.cb, 5)
            self.get_logger().info(f'saving every {args.every} frames from {args.topic} -> {args.out}')

        def cb(self, msg):
            self.n += 1
            if self.n % args.every:
                return
            try:
                from cv_bridge import CvBridge
                img = CvBridge().imgmsg_to_cv2(msg, 'bgr8')
            except Exception:
                a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
                img = a[:, :, ::-1] if msg.encoding == 'rgb8' else a
            path = os.path.join(args.out, f'frame_{int(time.time()*1000)}.jpg')
            cv2.imwrite(path, img)
            if self.n % (args.every * 20) == 0:
                self.get_logger().info(f'saved {self.n // args.every} images')

    rclpy.init()
    try:
        rclpy.spin(Collector())
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
