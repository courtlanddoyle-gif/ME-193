"""
Swing detection from the Double Motor's IMU.

The hub streams accelerometer (ax, ay, az) and gyroscope (gx, gy, gz)
readings. One reading on its own can't tell a swing from a bump, so we look
at a short window (the last SWING_WINDOW_S seconds) and boil it down to a
handful of summary numbers -- how hard it accelerated, how fast it rotated,
how sudden it was. A logistic regression classifier, trained on labeled
windows, turns those numbers into P(swing).

Units: we treat acceleration as milli-g (1000 = 1 g, so a hub sitting still
reads ~1000 total) and rotation as degrees/second. The hub's raw values may
use a different scale -- that's fine, because the classifier is retrained on
real hub data with collect_swing_data.py, and only the synthetic generator
below cares about the exact numbers.

The synthetic generator exists so the whole pipeline (collect -> train ->
test -> play) can be run without the hub. A model trained on synthetic data
proves the code works; it does not prove it will work on a real swing.
"""

import math
import threading
from collections import deque

import joblib
import numpy as np

import config

FEATURE_NAMES = [
    "acc_dyn_max",     # biggest departure from 1 g (how hard it was shoved)
    "acc_dyn_mean",
    "acc_mag_std",
    "acc_jerk_max",    # biggest sample-to-sample change in |acc| (suddenness)
    "gyro_mag_max",    # fastest rotation
    "gyro_mag_mean",
    "gyro_mag_std",
    "gyro_x_range",
    "gyro_y_range",
    "gyro_z_range",
]

GRAVITY_MG = 1000.0


def window_features(samples):
    """samples: array-like of shape (N, 6) = ax, ay, az, gx, gy, gz.
    Returns a 1-D feature vector, or None if there aren't enough samples."""
    s = np.asarray(samples, dtype=np.float64)
    if s.ndim != 2 or s.shape[0] < 3:
        return None
    acc, gyro = s[:, :3], s[:, 3:6]
    acc_mag = np.linalg.norm(acc, axis=1)
    acc_dyn = np.abs(acc_mag - GRAVITY_MG)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    return np.array([
        acc_dyn.max(),
        acc_dyn.mean(),
        acc_mag.std(),
        np.abs(np.diff(acc_mag)).max(),
        gyro_mag.max(),
        gyro_mag.mean(),
        gyro_mag.std(),
        *np.ptp(gyro, axis=0),
    ])


# --------------------------------------------------------------------------
# Synthetic IMU data (used by --sim mode and train_swing_classifier.py --synthetic)
# --------------------------------------------------------------------------

def random_swing(t0, rng):
    """One swing: a short bell-shaped burst of rotation + acceleration."""
    axis = rng.normal(size=3)
    axis[rng.integers(3)] += 2.5          # swings rotate mostly about one axis
    axis /= np.linalg.norm(axis)
    return dict(
        t0=t0,
        sigma=rng.uniform(0.045, 0.09),   # seconds; a real swing is ~0.2-0.3 s long
        gyro_peak=rng.uniform(250, 900),  # deg/s
        acc_peak=rng.uniform(700, 2500),  # milli-g on top of gravity
        axis=axis,
        acc_dir=rng.normal(size=3) / math.sqrt(3),
    )


def synthetic_sample(t, gravity_dir, swings, rng, wobble=None):
    """One IMU reading at time t.

    gravity_dir: unit vector for how the hub is being held.
    swings: list of dicts from random_swing().
    wobble: optional (amplitude_dps, freq_hz, phase) for gentle,
            non-swing hand movement -- the "hard negatives" that keep the
            classifier from calling every twitch a swing.
    """
    acc = GRAVITY_MG * np.asarray(gravity_dir) + rng.normal(0, 15, 3)
    gyro = rng.normal(0, 3, 3)
    if wobble is not None:
        amp, freq, phase = wobble
        w = math.sin(2 * math.pi * freq * t + phase)
        gyro += amp * w * np.array([0.6, 0.3, 0.75])
        acc += 0.6 * amp * math.cos(2 * math.pi * freq * t + phase) * np.array([0.5, 0.7, 0.4])
    for sw in swings:
        pulse = math.exp(-0.5 * ((t - sw["t0"]) / sw["sigma"]) ** 2)
        if pulse > 1e-3:
            gyro += sw["gyro_peak"] * pulse * sw["axis"]
            acc += sw["acc_peak"] * pulse * sw["acc_dir"]
    return np.concatenate([acc, gyro])


def random_gravity(rng):
    g = rng.normal(size=3) + np.array([0, 0, 2.0])   # mostly upright, but not always
    return g / np.linalg.norm(g)


def synthetic_window(label, rng, rate_hz=config.IMU_RATE_HZ, window_s=config.SWING_WINDOW_S):
    """Make one labeled window: 1 = a swing peaks inside it, 0 = it doesn't."""
    n = max(3, int(round(window_s * rate_hz)))
    times = np.sort(rng.uniform(0, window_s, n))     # BLE timing is jittery
    gravity = random_gravity(rng)
    swings = []
    wobble = None
    if label == 1:
        # the peak can land anywhere from mid-window to the very end, since
        # live detection only sees the swing up to "now"
        swings.append(random_swing(rng.uniform(0.4, 1.0) * window_s, rng))
    else:
        kind = rng.choice(["still", "wobble", "tail"], p=[0.35, 0.45, 0.2])
        if kind == "wobble":
            wobble = (rng.uniform(15, 110), rng.uniform(0.5, 2.5), rng.uniform(0, 2 * math.pi))
        elif kind == "tail":
            # the fading end of a swing that peaked well before this window
            sw = random_swing(-rng.uniform(0.25, 0.45), rng)
            swings.append(sw)
    return np.array([synthetic_sample(t, gravity, swings, rng, wobble) for t in times])


def synthetic_dataset(n_per_class=600, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label in (0, 1):
        for _ in range(n_per_class):
            X.append(window_features(synthetic_window(label, rng)))
            y.append(label)
    return np.array(X), np.array(y)


# --------------------------------------------------------------------------
# Live detector
# --------------------------------------------------------------------------

class SwingDetector:
    """Feeds the latest IMU window to the trained classifier and turns its
    probability into a clean swing event (latched briefly, then cooled down
    so one swing can't hit two balls)."""

    def __init__(self, model_path=config.SWING_MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"No swing model at {model_path}.\n"
                "Train one first:  python train_swing_classifier.py --synthetic\n"
                "(or record real hub data with collect_swing_data.py and train on that)"
            )
        self.model = joblib.load(model_path)
        self.prob = 0.0
        self._latched_until = 0.0
        self._cooldown_until = 0.0

    def update(self, now, window):
        feats = window_features(window)
        if feats is None:
            self.prob = 0.0
            return
        self.prob = float(self.model.predict_proba(feats.reshape(1, -1))[0, 1])
        if self.prob >= config.SWING_THRESHOLD and now >= self._cooldown_until:
            self._latched_until = now + config.SWING_LATCH_S

    def swinging(self, now):
        return now < self._latched_until

    def consume(self, now):
        """Call when a swing hits the ball, so the same swing can't count twice."""
        self._latched_until = 0.0
        self._cooldown_until = now + config.SWING_COOLDOWN_S


class ImuBuffer:
    """Rolling buffer of (timestamp, 6 IMU values). The Bluetooth thread
    writes while the game thread reads, so every access takes a lock --
    without it, reading mid-write can raise "deque mutated during iteration"."""

    def __init__(self, seconds=2.0):
        self._buf = deque()
        self._seconds = seconds
        self._lock = threading.Lock()

    def add(self, t, sample):
        with self._lock:
            self._buf.append((t, np.asarray(sample, dtype=np.float64)))
            while self._buf and self._buf[0][0] < t - self._seconds:
                self._buf.popleft()

    def since(self, t):
        """All (timestamp, sample) pairs newer than t."""
        with self._lock:
            return [(ts, s) for (ts, s) in self._buf if ts > t]

    def window(self, now, window_s=config.SWING_WINDOW_S):
        with self._lock:
            rows = [s for (t, s) in self._buf if t >= now - window_s]
        return np.array(rows) if rows else np.empty((0, 6))
