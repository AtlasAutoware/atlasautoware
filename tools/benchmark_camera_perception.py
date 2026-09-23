#!/usr/bin/env python3
"""Benchmark the same direct TensorRT detector used by the ROS node."""

import argparse
import os
import statistics
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'f1tenth_gym_ros'))
from camera_perception import TRTDetector                    # noqa: E402


def percentile(values, fraction):
    """Return a nearest-rank percentile from a non-empty sorted sequence."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('engine', help='TensorRT .engine built on this Jetson')
    parser.add_argument(
        '--image',
        help='optional camera image; otherwise a blank 640x480 frame')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument(
        '--confidence', type=float, default=0.50,
        help='detection threshold (default: hardware.yaml value, 0.50)')
    args = parser.parse_args()

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            raise SystemExit(f'could not read image: {args.image}')
    else:
        image = np.zeros((480, 640, 3), np.uint8)

    detector = TRTDetector(args.engine, conf=args.confidence)
    try:
        for _ in range(max(0, args.warmup)):
            detector.detect(image)
        samples = []
        boxes = []
        for _ in range(max(1, args.runs)):
            started = time.perf_counter()
            boxes = detector.detect(image)
            samples.append((time.perf_counter() - started) * 1000.0)
    finally:
        detector.close()

    mean_ms = statistics.fmean(samples)
    print(f'backend=tensorrt input={detector.sz} runs={len(samples)}')
    print(f'end_to_end_ms mean={mean_ms:.2f} '
          f'p50={percentile(samples, 0.50):.2f} '
          f'p95={percentile(samples, 0.95):.2f} '
          f'fps={1000.0 / mean_ms:.1f}')
    print(f'detections={len(boxes)}')
    if args.image:
        for index, (x, y, width, height, confidence) in enumerate(boxes):
            print(f'box[{index}] x={x:.1f} y={y:.1f} w={width:.1f} '
                  f'h={height:.1f} conf={confidence:.4f}')


if __name__ == '__main__':
    main()
