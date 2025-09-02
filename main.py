#!/usr/bin/env python3

import sys
import time
import argparse
from datetime import datetime
from ServoMotor import ServoMotor
import numpy as np
from picamera2 import Picamera2
import cv2


WIDTH = 640
HEIGHT = 480

servo_motor = ServoMotor()


def get_frame_from_picamera2(args):
    """Initialize Picamera2 and yield frames as numpy arrays in BGR order."""
    picam2 = Picamera2()

    # Configure preview/stream size
    width = args.width
    height = args.height

    # Create a simple preview configuration in RGB888
    config = picam2.create_preview_configuration(
        main={"size": (width, height), "format": "RGB888"}
    )
    picam2.configure(config)
    # picam2.set_controls({"FrameRate": 1.0})
    picam2.start()

    try:
        while True:
            frame = picam2.capture_array()
            yield frame
    finally:
        picam2.stop()


def motion_detector(args):
    # Select frame source
    frame_source = get_frame_from_picamera2(args)
    backend = "picamera2"

    print(f"[INFO] Using backend: {backend}")
    print(
        f"[INFO] min_area={args.min_area}, cooldown={args.cooldown}s, show={args.show}")

    # Background subtractor
    backsub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=True)

    # Morphology kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    last_event_ts = 0.0

    for frame in frame_source:
        now = time.time()
        if (now - last_event_ts) < args.cooldown:
            continue
        # Process frame
        fgmask = backsub.apply(frame)
        # Morphology to clean noise
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
        fgmask = cv2.dilate(fgmask, kernel, iterations=2)
        # Threshold to binary
        _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_detected = False
        biggest_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            print(area)
            if area > args.min_area:
                motion_detected = True
                if area > biggest_area:
                    biggest_area = area

        if motion_detected:
            last_event_ts = now
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[MOTION] {timestamp} | area≈{int(biggest_area)}")
            cv2.imwrite(f"{timestamp}.jpg", frame)



def parse_args():
    p = argparse.ArgumentParser(
        description="Simple motion detection from Raspberry Pi camera.")
    p.add_argument("--min-area", type=int, default=1000,
                   help="Minimum contour area to consider as motion (higher = less sensitive).")
    p.add_argument("--cooldown", type=float, default=1.0,
                   help="Seconds to wait between motion prints (debounce).")
    p.add_argument("--show", action="store_true",
                   help="Show live preview and motion mask (GUI).")
    p.add_argument("--width", type=int, default=640, help="Frame width.")
    p.add_argument("--height", type=int, default=480, help="Frame height.")
    p.add_argument("--open", type=int, default=0, help="open")
    p.add_argument("--close", type=int, default=0, help="open")
    return p.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        motion_detector(args)

        print(args.open)
        if args.open == 1:
            servo_motor.open()
        elif args.close == 1:
            servo_motor.close()

    except KeyboardInterrupt:
        servo_motor.pi.stop()
        print("\n[INFO] Exiting (Ctrl+C).")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
