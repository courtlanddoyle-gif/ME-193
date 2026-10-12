"""
Make the AprilTags the game looks for (tag36h11 family) as PNGs in tags/,
plus one sheet with all four. Print the sheet, or open a single tag on your
phone and hold the screen up to the camera.

    python make_tags.py
"""

import cv2
import numpy as np

import config

OUT = config.HERE / "tags"
SIZE = 400          # pixels per tag
BORDER = 60         # white margin -- the detector needs it


def tag_image(tag_id, label):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag = cv2.aruco.generateImageMarker(dictionary, tag_id, SIZE)
    img = np.full((SIZE + 2 * BORDER + 50, SIZE + 2 * BORDER), 255, np.uint8)
    img[BORDER:BORDER + SIZE, BORDER:BORDER + SIZE] = tag
    cv2.putText(img, f"{tag_id}: {label}", (BORDER, SIZE + BORDER + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
    return img


def main():
    OUT.mkdir(exist_ok=True)
    labels = {config.START_TAG_ID: "START", **{k: v.upper() for k, v in config.LEVEL_TAG_IDS.items()}}
    images = []
    for tag_id, label in sorted(labels.items()):
        img = tag_image(tag_id, label)
        path = OUT / f"tag_{tag_id}_{label.lower()}.png"
        cv2.imwrite(str(path), img)
        images.append(img)
        print(f"Wrote {path}")
    sheet = np.vstack([np.hstack(images[:2]), np.hstack(images[2:4])])
    cv2.imwrite(str(OUT / "all_tags_sheet.png"), sheet)
    print(f"Wrote {OUT / 'all_tags_sheet.png'}")


if __name__ == "__main__":
    main()
