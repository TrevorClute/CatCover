#!/usr/bin/env python3

import sys
import time
import argparse
from datetime import datetime
from ServoMotor import ServoMotor
from NcnnModel import NcnnNodel
import numpy as np
from picamera2 import Picamera2
import cv2

from libcamera import Transform


WIDTH = 640
HEIGHT = 480
IMAGE_SIZE = 640

servo_motor = ServoMotor()
ncnn_model = NcnnNodel()


def get_frame_from_picamera2(args):
    """Initialize Picamera2 and yield frames as numpy arrays in BGR order."""
    picam2 = Picamera2()

    # Configure preview/stream size
    # width = args.width
    # height = args.height

    # Create a simple preview configuration in RGB888
    # config = picam2.create_preview_configuration(
    #     main={"size": (width, height), "format": "RGB888"}
    # )
    # 1) Pick the largest native sensor mode (max area = width*height)
    best = max(picam2.sensor_modes, key=lambda m: m["size"][0] * m["size"][1])
    W, H = best["size"]

    # 2) Configure video with the sensor’s full aspect ratio & no crop
    #    - Use the mode's native size to avoid aspect-crop
    #    - ScalerCrop set to the full frame (no digital zoom)
    config = picam2.create_video_configuration(
        main={"size": (W, H), "format": "RGB888"},   # or "XBGR8888" if you prefer
        transform=Transform(),                       # no flips/rotations
        buffer_count=4,
        controls={"ScalerCrop": (0, 0, W, H)}        # full sensor area
    )

    # config = picam2.create_video_configuration(
    #     main={"size": (640, 480), "format": "RGB888"}
    # )
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
        f"[INFO] min_area={args.min_area}, cooldown={args.cooldown}s")

    # Background subtractor
    backsub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=False)

    # Morphology kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    last_event_ts = 0.0

    for frame in frame_source:
        now = time.time()
        fgmask = backsub.apply(frame)
        if (now - last_event_ts) < args.cooldown:
            continue

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
            if area > args.min_area:
                motion_detected = True
                if area > biggest_area:
                    biggest_area = area

        if motion_detected:
            last_event_ts = now
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            predictions = ncnn_model.predict(frame)

            if predictions[0]["name"] == "marbles":
                # servo_motor.close()
                cv2.imwrite(f"imgs/m {predictions[0]['conf']} -- {timestamp}.jpg", frame)
                args.cooldown = 0
            elif predictions[0]["name"] == "teddy_or_jesse":
                # servo_motor.open()
                cv2.imwrite(f"imgs/tj {predictions[0]['conf']} -- {timestamp}.jpg", frame)
                args.cooldown = 0.5
            else:
                args.cooldown = 0.3




def parse_args():
    p = argparse.ArgumentParser(
        description="Simple motion detection from Raspberry Pi camera.")
    p.add_argument("--min-area", type=int, default=3500,
                   help="Minimum contour area to consider as motion (higher = less sensitive).")
    p.add_argument("--cooldown", type=float, default=0.0,
                   help="Seconds to wait between motion prints (debounce).")
    p.add_argument("--width", type=int, default=640, help="Frame width.")
    p.add_argument("--height", type=int, default=480, help="Frame height.")
    p.add_argument("--open", type=int, default=0, help="open")
    p.add_argument("--close", type=int, default=0, help="open")
    return p.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        motion_detector(args)

    except KeyboardInterrupt:
        servo_motor.pi.stop()
        print("\n[INFO] Exiting (Ctrl+C).")
    except Exception as e:
        print(f"[ERROR] {e.with_traceback()}")
        sys.exit(1)
