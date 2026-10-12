"""
Fine-tune YOLOv8n to find the LEGO Double Motor, from the photos labeled
with label_hub_photos.py. The game automatically uses the result
(models/hub_yolo.pt) instead of the generic pretrained model.

    python train_hub_yolo.py
    python train_hub_yolo.py --epochs 100

Transfer learning: we start from yolov8n.pt, which already learned general
shapes and edges from the COCO dataset, and only teach it one new class.
That's why ~15 photos can be enough. YOLO's built-in augmentation (random
flips, scaling, color shifts, mosaics) stretches those 15 photos further.
"""

import argparse
import shutil

import config

DATA_YAML = config.DATA_DIR / "hub_yolo" / "data.yaml"
RUNS = config.HERE / "runs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit("No dataset yet -- run label_hub_photos.py first.")

    from ultralytics import YOLO

    model = YOLO(config.COCO_YOLO_WEIGHTS)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=8,
        patience=25,          # stop early if validation stops improving
        project=str(RUNS),
        name="hub",
        exist_ok=True,
        plots=True,
        device="cpu",
        workers=0,            # Windows-friendly
        verbose=False,
    )

    best = RUNS / "hub" / "weights" / "best.pt"
    config.MODELS_DIR.mkdir(exist_ok=True)
    shutil.copy(best, config.HUB_YOLO_WEIGHTS)
    print(f"\nCopied {best} -> {config.HUB_YOLO_WEIGHTS}")

    metrics = YOLO(str(config.HUB_YOLO_WEIGHTS)).val(data=str(DATA_YAML), device="cpu",
                                                      workers=0, verbose=False, plots=False)
    print(f"Validation  precision {metrics.box.mp:.2f}   recall {metrics.box.mr:.2f}   "
          f"mAP50 {metrics.box.map50:.2f}")
    print("Training curves and example predictions are in runs/hub/ (good for the write-up).")
    print("The game will now use this model automatically.")


if __name__ == "__main__":
    main()
