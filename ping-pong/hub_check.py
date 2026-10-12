"""
First thing to run with the real Double Motor. Checks the connection,
shows what the IMU numbers actually look like, and plays each haptic
pattern so you can feel them.

    python hub_check.py

Follow the prompts: hold still, then swing a few times. Paste the summary
it prints at the end back to Claude -- it tells us whether the IMU units
match what the synthetic swing data assumed (1000 = 1 g, deg/s).
"""

import time

import numpy as np

from paddle_hw import LegoPaddle


def record(paddle, seconds, prompt):
    input(f"\n{prompt}  Press Enter to start ({seconds} s)...")
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        time.sleep(0.05)
    data = np.array([s for _, s in paddle.imu_since(t0)])
    rate = len(data) / seconds
    acc_mag = np.linalg.norm(data[:, :3], axis=1) if len(data) else np.array([0.0])
    gyro_mag = np.linalg.norm(data[:, 3:], axis=1) if len(data) else np.array([0.0])
    print(f"  {len(data)} readings ({rate:.0f} per second)")
    print(f"  |acc|  mean {acc_mag.mean():8.1f}   min {acc_mag.min():8.1f}   max {acc_mag.max():8.1f}")
    print(f"  |gyro| mean {gyro_mag.mean():8.1f}   max {gyro_mag.max():8.1f}")
    return rate, acc_mag, gyro_mag


def main():
    paddle = LegoPaddle()
    try:
        time.sleep(0.5)
        still = record(paddle, 3, "Hold the hub STILL in your hand.")
        swing = record(paddle, 5, "Now SWING it like a paddle 4-5 times.")

        print("\nHaptics -- feel each one:")
        for event in ("serve", "hit", "record", "miss"):
            input(f"  Press Enter to play '{event}'...")
            paddle.haptic(event)
            time.sleep(1.5)

        print("\n===== SUMMARY (paste this to Claude) =====")
        print(f"IMU rate: {still[0]:.0f} Hz")
        print(f"still: |acc| mean {still[1].mean():.1f}, |gyro| mean {still[2].mean():.1f}")
        print(f"swing: |acc| max {swing[1].max():.1f}, |gyro| max {swing[2].max():.1f}")
        print("Assumed by the synthetic data: still |acc| ~1000, swing |gyro| max 250-900")
    finally:
        paddle.close()


if __name__ == "__main__":
    main()
