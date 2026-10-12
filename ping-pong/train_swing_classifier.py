"""
Train the swing / no-swing classifier.

    python train_swing_classifier.py --synthetic   # no hub needed
    python train_swing_classifier.py               # uses data/swing_data.csv

Cuts the recorded IMU stream into SWING_WINDOW_S windows, turns each into
the summary features in imu_swing.window_features(), fits a logistic
regression, and reports accuracy on a held-out 20% test split.

How windows get labels from collect_swing_data.py recordings:
  - SWING mode: find each swing's peak rotation speed, and make a few
    windows that end at / just after that peak       -> label 1
  - IDLE mode: slide a window across the whole thing -> label 0
"""

import argparse
import csv

import joblib
import numpy as np
from scipy.signal import find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from imu_swing import FEATURE_NAMES, synthetic_dataset, window_features

DATA_PATH = config.DATA_DIR / "swing_data.csv"
W = config.SWING_WINDOW_S


def load_recording(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["t"]) for r in rows])
    imu = np.array([[float(r[k]) for k in ("ax", "ay", "az", "gx", "gy", "gz")] for r in rows])
    mode = np.array([r["mode"] for r in rows])
    return t, imu, mode


def windows_from_recording(t, imu, mode):
    """Returns X, y and groups. Windows that overlap (the three around one
    swing, or neighbouring idle windows) share a group id, so the train/test
    split can keep them on the same side -- otherwise the test set contains
    near-copies of training windows and the accuracy looks better than it is."""
    X, y, groups = [], [], []

    def add(end_t, label, t_seg, imu_seg, group):
        sel = (t_seg > end_t - W) & (t_seg <= end_t)
        feats = window_features(imu_seg[sel])
        if feats is not None:
            X.append(feats)
            y.append(label)
            groups.append(group)

    # swing windows: one per peak, at a few positions so the model learns to
    # fire as soon as the peak arrives, not only once it's fully past
    sw = mode == "swing"
    if sw.any():
        ts, gs = t[sw], imu[sw]
        gyro_mag = np.linalg.norm(gs[:, 3:], axis=1)
        rate = len(ts) / max(1e-6, ts[-1] - ts[0])
        height = 0.5 * np.percentile(gyro_mag, 95)
        peaks, _ = find_peaks(gyro_mag, height=height, distance=max(1, int(0.4 * rate)))
        print(f"Found {len(peaks)} swings in SWING mode (peak threshold {height:.0f} deg/s)")
        for i, p in enumerate(peaks):
            for after in (0.0, 0.05, 0.1):
                add(ts[p] + after, 1, ts, gs, group=f"swing{i}")

    idle = mode == "idle"
    if idle.any():
        ti, gi = t[idle], imu[idle]
        for end_t in np.arange(ti[0] + W, ti[-1], 0.1):
            add(end_t, 0, ti, gi, group=f"idle{int((end_t - ti[0]) // 1.0)}")   # 1 s blocks

    return np.array(X), np.array(y), np.array(groups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                        help="train on generated data instead of a hub recording")
    args = parser.parse_args()

    if args.synthetic:
        print("Training on SYNTHETIC IMU data (pipeline test, not real-world accuracy).")
        X, y = synthetic_dataset()
        groups = None   # every synthetic window is independent
    else:
        if not DATA_PATH.exists():
            raise SystemExit(f"No recording at {DATA_PATH}. Run collect_swing_data.py first, "
                             "or use --synthetic.")
        X, y, groups = windows_from_recording(*load_recording(DATA_PATH))

    print(f"{len(y)} windows: {int((y == 1).sum())} swing, {int((y == 0).sum())} not-swing")
    if len(set(y)) < 2:
        raise SystemExit("Need both swing and idle data to train.")

    if groups is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0, stratify=y)
    else:
        split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        train_i, test_i = next(split.split(X, y, groups))
        X_train, X_test, y_train, y_test = X[train_i], X[test_i], y[train_i], y[test_i]
        print(f"Grouped split: {len(set(groups[test_i]))} held-out swings/idle-seconds "
              "never seen in training")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print(f"\nTest accuracy: {accuracy_score(y_test, pred):.3f}")
    print("Confusion matrix (rows = true, cols = predicted; order: not-swing, swing):")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=["not-swing", "swing"]))

    weights = model.named_steps["logisticregression"].coef_[0]
    print("Feature weights (on standardized features; bigger = more 'swing-like'):")
    for name, w in sorted(zip(FEATURE_NAMES, weights), key=lambda p: -abs(p[1])):
        print(f"  {name:>14}: {w:+.2f}")

    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, config.SWING_MODEL_PATH)
    print(f"\nSaved model to {config.SWING_MODEL_PATH}")


if __name__ == "__main__":
    main()
