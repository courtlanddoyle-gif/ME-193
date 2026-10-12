"""
Turn a Roboflow YOLOv8 export into the hub dataset train_hub_yolo.py uses
(the alternative to labeling with label_hub_photos.py).

    python import_roboflow.py                         # finds "*.yolov8" here
    python import_roboflow.py "Ping Pong Training.yolov8"

What it does:
  - Outlines (polygons) become boxes, since we train a box detector. There's
    one hub per photo, so if the hub was outlined in pieces (e.g. split by
    your fingers) the pieces are merged into one box.
  - Labels with fewer than 3 points aren't outlines -- usually a stray click
    -- so those photos are left out and listed. Fix them in Roboflow and
    re-export to get them back.
  - A photo with an empty label file is kept as a "no hub here" example.
  - Roboflow put everything in train/, so ~20% of the hub photos are held
    out for validation. Without that there's no honest accuracy number.
  - Big photos are shrunk to 1280 px (YOLO trains at 640 anyway).
"""

import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

import config

DATASET = config.DATA_DIR / "hub_yolo"
MAX_SIDE = 1280
VAL_FRACTION = 0.2


def find_export():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    exports = sorted(p for p in config.HERE.glob("*.yolov8") if p.is_dir())
    if not exports:
        raise SystemExit("No Roboflow export folder (*.yolov8) found -- unzip it here or pass its path.")
    return exports[-1]


def label_to_box(label_path, polygon_export):
    """Returns ("box", line), ("empty", None) or ("broken", reason).

    A box line and a 2-point polygon both have 4 numbers, so which one a
    4-number line is depends on the export: in a polygon export it's a
    broken outline (usually a stray click), not a box."""
    lines = [l.split() for l in label_path.read_text().splitlines() if l.strip()]
    if not lines:
        return "empty", None
    points = []
    for v in lines:
        coords = np.array(v[1:], dtype=float)
        if len(coords) == 4 and not polygon_export:    # a box: cx cy w h
            cx, cy, w, h = coords
            coords = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        pts = coords.reshape(-1, 2)
        if polygon_export and len(pts) < 3:
            return "broken", f"an outline with only {len(pts)} point(s)"
        points.append(pts)
    pts = np.clip(np.vstack(points), 0, 1)
    (x1, y1), (x2, y2) = pts.min(0), pts.max(0)
    return "box", f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"


def main():
    export = find_export()
    print(f"Importing {export}")
    label_files = list(export.glob("*/labels/*.txt"))
    polygon_export = any(len(l.split()) > 5 for f in label_files for l in f.read_text().splitlines())
    print(f"Label format: {'outlines (polygons)' if polygon_export else 'boxes'}")
    items, broken = [], []
    for split_dir in ("train", "valid", "test"):
        for img_path in sorted((export / split_dir / "images").glob("*")):
            label_path = export / split_dir / "labels" / (img_path.stem + ".txt")
            if not label_path.exists():
                continue
            kind, info = label_to_box(label_path, polygon_export)
            if kind == "broken":
                broken.append((img_path.name, info))
                continue
            items.append((img_path, info))

    hub = [x for x in items if x[1] is not None]
    background = [x for x in items if x[1] is None]
    random.Random(0).shuffle(hub)
    n_val = max(1, round(VAL_FRACTION * len(hub)))
    splits = {"val": hub[:n_val], "train": hub[n_val:] + background}

    if DATASET.exists():
        shutil.rmtree(DATASET)
    for split, rows in splits.items():
        (DATASET / "images" / split).mkdir(parents=True)
        (DATASET / "labels" / split).mkdir(parents=True)
        for img_path, line in rows:
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            scale = min(1.0, MAX_SIDE / max(h, w))
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            stem = img_path.stem.split("_jpg.rf.")[0]     # drop Roboflow's random suffix
            cv2.imwrite(str(DATASET / "images" / split / f"{stem}.jpg"), img)
            (DATASET / "labels" / split / f"{stem}.txt").write_text(line + "\n" if line else "")

    (DATASET / "data.yaml").write_text(
        f"path: {DATASET.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: lego_hub\n")

    print(f"{len(hub)} photos with the hub, {len(background)} without")
    print(f"  train: {len(splits['train'])}   val: {len(splits['val'])}   -> {DATASET}")
    for name, why in broken:
        print(f"  LEFT OUT {name}: {why} -- redraw it in Roboflow and re-export")
    print("Next:  python train_hub_yolo.py")


if __name__ == "__main__":
    main()
