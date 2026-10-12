"""
Record labeled IMU data for the swing classifier.

    python collect_swing_data.py          # real hub
    python collect_swing_data.py --sim    # dry run of the pipeline, no hub

Keys (click the window first):
    1  start recording SWING mode  -- swing the hub like a paddle, over and
       over, with a short pause between swings. Mix forehands and backhands.
    2  start recording IDLE mode   -- hold the hub and move around normally
       (reposition, walk, raise/lower it) but DON'T swing. These are the
       examples that teach the model what a swing is not.
    0  pause recording
    SPACE  (sim only) fake a swing
    q  save to data/swing_data.csv and quit

Aim for ~40 swings and ~60 seconds of idle movement. Each row saved is one
raw IMU reading; train_swing_classifier.py cuts them into windows.
"""

import argparse
import csv
import time

import cv2
import numpy as np

import config
from paddle_hw import make_paddle

OUT_PATH = config.DATA_DIR / "swing_data.csv"
MODES = {ord("1"): "swing", ord("2"): "idle", ord("0"): None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="use the simulated paddle")
    args = parser.parse_args()

    paddle = make_paddle(args.sim)
    mode = None
    rows = []
    last_saved_t = 0.0
    next_auto_swing = 0.0
    history = []   # recent gyro magnitudes for the live plot

    try:
        while True:
            now = time.monotonic()
            window = paddle.imu_window(now)
            if len(window):
                history.append(float(np.linalg.norm(window[-1, 3:])))
                history = history[-300:]
            if mode is not None:
                # save every reading newer than the last one we saved
                for t, s in paddle.imu_since(last_saved_t):
                    rows.append([t, *s, mode])
                    last_saved_t = t
            else:
                last_saved_t = now

            if args.sim and mode == "swing" and now > next_auto_swing:
                paddle.swing()
                next_auto_swing = now + np.random.uniform(0.9, 1.6)

            img = np.zeros((240, 600, 3), np.uint8)
            if len(history) > 1:
                pts = np.array([[i * 2, 230 - min(220, h / 4)] for i, h in enumerate(history)], np.int32)
                cv2.polylines(img, [pts], False, (0, 220, 255), 1)
            label = f"mode: {mode or 'paused'}   rows: {len(rows)}"
            cv2.putText(img, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(img, "1=swing  2=idle  0=pause  q=save", (10, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            cv2.imshow("Collect swing data (gyro magnitude)", img)

            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break
            if key in MODES:
                mode = MODES[key]
                print(f"Mode -> {mode or 'paused'}")
            if key == ord(" "):
                paddle.swing()
    finally:
        paddle.close()
        cv2.destroyAllWindows()

    config.DATA_DIR.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "ax", "ay", "az", "gx", "gy", "gz", "mode"])
        writer.writerows(rows)
    print(f"Saved {len(rows)} readings to {OUT_PATH}")


if __name__ == "__main__":
    main()
