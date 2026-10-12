"""
Build the YOLO training set for the hub: draw a box around the Double
Motor in each photo. (Same idea as Roboflow for the blue-minifig project,
but built in, so there's nothing to upload or export.)

Two ways to get photos:

    python label_hub_photos.py                 # label photos in hub_photos/
    python label_hub_photos.py --capture       # take them with the webcam

--capture is the better choice: the photos come from the same camera, at
the same distance and lighting, as the game. Press SPACE to snap a photo,
q when done, and then you label them.

Labeling keys:
    drag with the mouse   draw the box around the hub (redraw to fix it)
    ENTER / SPACE         save this box, next photo
    n                     no hub in this photo (saved as a background example)
    s                     skip this photo
    q                     stop

Tips for ~15 good photos: hold the hub the way you play, vary where it is
in the frame (left/right/high/low), its angle, how close you are, and the
lighting. Make the box tight around the hub body, not your hand.

Output: data/hub_yolo/ in YOLO format (images/, labels/, data.yaml), split
into train and val, ready for train_hub_yolo.py.
"""

import argparse
import random
import shutil

import cv2

import config

PHOTO_DIR = config.HERE / "hub_photos"
DATASET = config.DATA_DIR / "hub_yolo"
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_SIDE = 1280      # shrink big phone photos; YOLO trains at 640 anyway
SHOW_SIDE = 900      # how big to show photos while labeling
WIN = "Label the hub (drag a box, ENTER = save, n = no hub, s = skip, q = quit)"


def capture_photos():
    PHOTO_DIR.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    n = len(list(PHOTO_DIR.glob("cam_*.jpg")))
    print("SPACE = take photo, q = done")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        preview = cv2.flip(frame, 1)
        cv2.putText(preview, f"{n} photos   SPACE = snap   q = done", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Capture hub photos", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            cv2.imwrite(str(PHOTO_DIR / f"cam_{n:03d}.jpg"), frame)   # un-mirrored, like the game sees it
            n += 1
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def load_photo(path):
    img = cv2.imread(str(path))    # applies phone EXIF rotation
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def label_one(img, name, index, total):
    """Returns ("box", (x1, y1, x2, y2)) in image pixels, ("none", None),
    ("skip", None) or ("quit", None)."""
    h, w = img.shape[:2]
    scale = SHOW_SIDE / max(h, w)
    shown = cv2.resize(img, (int(w * scale), int(h * scale)))
    state = {"start": None, "box": None, "dragging": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state.update(start=(x, y), box=None, dragging=True)
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["box"] = (*state["start"], x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["dragging"]:
            state["dragging"] = False
            state["box"] = (*state["start"], x, y)

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)
    while True:
        view = shown.copy()
        if state["box"]:
            x1, y1, x2, y2 = state["box"]
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(view, f"{index + 1}/{total}  {name}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(WIN, view)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, ord(" ")) and state["box"]:
            x1, y1, x2, y2 = state["box"]
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            return "box", tuple(int(round(v / scale)) for v in (x1, y1, x2, y2))
        if key == ord("n"):
            return "none", None
        if key == ord("s"):
            return "skip", None
        if key == ord("q"):
            return "quit", None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true", help="take photos with the webcam first")
    parser.add_argument("--val", type=float, default=0.2, help="fraction of photos held out for validation")
    args = parser.parse_args()

    if args.capture:
        capture_photos()

    photos = sorted(p for p in PHOTO_DIR.glob("*") if p.suffix.lower() in EXTS) if PHOTO_DIR.exists() else []
    if not photos:
        raise SystemExit(f"No photos found. Put JPG/PNG photos in {PHOTO_DIR} "
                         "(iPhone HEIC photos need converting to JPG first), or use --capture.")

    labeled = []   # (image, yolo_line or None, stem)
    for i, path in enumerate(photos):
        img = load_photo(path)
        if img is None:
            print(f"Can't read {path.name} -- skipping (HEIC? convert it to JPG)")
            continue
        action, box = label_one(img, path.name, i, len(photos))
        if action == "quit":
            break
        if action == "skip":
            continue
        line = None
        if action == "box":
            h, w = img.shape[:2]
            x1, y1, x2, y2 = box
            line = f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
        labeled.append((img, line, path.stem))
    cv2.destroyAllWindows()

    with_hub = [x for x in labeled if x[1] is not None]
    if len(with_hub) < 4:
        raise SystemExit(f"Only {len(with_hub)} photos labeled with the hub -- need at least 4.")

    # fresh dataset every time, so relabeling replaces the old labels
    if DATASET.exists():
        shutil.rmtree(DATASET)
    random.Random(0).shuffle(labeled)
    n_val = max(1, round(args.val * len(labeled)))
    splits = {"val": labeled[:n_val], "train": labeled[n_val:]}
    for split, items in splits.items():
        (DATASET / "images" / split).mkdir(parents=True)
        (DATASET / "labels" / split).mkdir(parents=True)
        for img, line, stem in items:
            cv2.imwrite(str(DATASET / "images" / split / f"{stem}.jpg"), img)
            (DATASET / "labels" / split / f"{stem}.txt").write_text(line + "\n" if line else "")

    (DATASET / "data.yaml").write_text(
        f"path: {DATASET.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: lego_hub\n")
    print(f"\nSaved {len(labeled)} photos ({len(with_hub)} with the hub): "
          f"{len(splits['train'])} train, {len(splits['val'])} val -> {DATASET}")
    print("Next:  python train_hub_yolo.py")


if __name__ == "__main__":
    main()
