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


WIDTH = 640
HEIGHT = 480
IMAGE_SIZE = 640

servo_motor = ServoMotor()
ncnn_model = NcnnNodel()


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
        f"[INFO] min_area={args.min_area}, cooldown={args.cooldown}s")

    # Background subtractor
    backsub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=True)

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
            print(f"[MOTION] {timestamp} | area≈{int(biggest_area)}")
            cv2.imwrite(f"imgs/{timestamp}.jpg",frame)
            # predictions = ncnn_model.predict(frame)
            # print(predictions)
            # highest_conf = max(
            #         predictions, key=lambda d: d.get("conf"), default={"conf":0, "name":""})
            # highest_conf_name = highest_conf.get("name")
            # if highest_conf_name == "marbles":
            #     servo_motor.close()
            # else:
            #     servo_motor.open()


def parse_args():
    p = argparse.ArgumentParser(
        description="Simple motion detection from Raspberry Pi camera.")
    p.add_argument("--min-area", type=int, default=2000,
                   help="Minimum contour area to consider as motion (higher = less sensitive).")
    p.add_argument("--cooldown", type=float, default=0.2,
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
