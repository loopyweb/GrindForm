"""Webcam pushup form checker. Run `python main.py --help` for options; press q to quit."""
import argparse
import os
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from form_logic import MSG_NOT_VISIBLE, PushupCounter, side_points

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pose_landmarker_lite.task")

WHITE = (255, 255, 255)
RED = (0, 0, 255)
ORANGE = (245, 117, 16)


def ensure_model(path):
    if os.path.exists(path):
        return
    print(f"Downloading pose model to {path} ...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".part"
    urllib.request.urlretrieve(MODEL_URL, tmp_path)
    os.replace(tmp_path, path)


def create_landmarker(model_path):
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def draw_overlay(frame, counter):
    cv2.rectangle(frame, (0, 0), (250, 80), ORANGE, -1)
    cv2.putText(frame, f"REPS: {counter.reps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)
    cv2.putText(frame, f"STAGE: {counter.stage or '-'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)
    cv2.putText(frame, counter.feedback, (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)


def run(source, model_path):
    ensure_model(model_path)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: could not open video source {source!r}", file=sys.stderr)
        return 1

    counter = PushupCounter()
    last_timestamp = -1
    with create_landmarker(model_path) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # VIDEO mode needs strictly increasing timestamps.
            timestamp = max(int(time.monotonic() * 1000), last_timestamp + 1)
            last_timestamp = timestamp
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                height, width = frame.shape[:2]
                points = side_points(landmarks, width, height)
                if points:
                    counter.update_from_points(points)
                else:
                    counter.feedback = MSG_NOT_VISIBLE
                vision.drawing_utils.draw_landmarks(
                    frame, landmarks, vision.PoseLandmarksConnections.POSE_LANDMARKS)

            draw_overlay(frame, counter)
            cv2.imshow("Pushup Form Checker", frame)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Total reps: {counter.reps}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Count pushups and check your form with a webcam.")
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument("--video", help="analyze a video file instead of the webcam")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="path to the pose landmarker .task model")
    args = parser.parse_args(argv)
    return run(args.video if args.video else args.camera, args.model)


if __name__ == "__main__":
    sys.exit(main())
