"""
Everything the camera tells the game:

    PoseTracker  -- MediaPipe's pretrained Pose Landmarker -> where your
                    paddle-hand wrist is.
    HubDetector  -- YOLO -> where the hub (paddle) itself is. Uses custom
                    weights trained on hub photos if models/hub_yolo.pt
                    exists, otherwise the pretrained COCO model looking for
                    things a hand-held hub/paddle gets detected as.
    TagReader    -- OpenCV's AprilTag detector -> which level/start tag
                    you're holding up.
    PaddleLocator -- fuses the two position estimates into one paddle x.

All detection runs on the UN-mirrored camera frame (AprilTags can't be
decoded in a mirror image). Positions are returned in normalized image
coordinates (0..1), and the game mirrors x for display so the screen moves
the way you do.
"""

import ssl
import urllib.request

import cv2
import numpy as np

import config

POSE_MODEL_PATH = config.MODELS_DIR / "pose_landmarker_lite.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
WRIST = {"left": 15, "right": 16}
ELBOW = {"left": 13, "right": 14}


def _download_pose_model():
    if POSE_MODEL_PATH.exists():
        return
    import certifi
    POSE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pretrained pose model to {POSE_MODEL_PATH} ...")
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(POSE_MODEL_URL, context=ctx) as response:
        POSE_MODEL_PATH.write_bytes(response.read())


class PoseTracker:
    def __init__(self, hand=config.PADDLE_HAND):
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        _download_pose_model()
        self._mp = mp
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH),
                                     delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self.hand = hand
        self.landmarks = None

    def process(self, frame_bgr, timestamp_ms):
        """Returns the paddle-hand wrist as (x, y) in 0..1, or None."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.pose_landmarks:
            self.landmarks = None
            return None
        self.landmarks = result.pose_landmarks[0]
        w = self.landmarks[WRIST[self.hand]]
        if getattr(w, "visibility", 1.0) < 0.3:
            return None
        return (w.x, w.y)

    def close(self):
        self._landmarker.close()


class HubDetector:
    def __init__(self):
        from ultralytics import YOLO

        if config.HUB_YOLO_WEIGHTS.exists():
            self.model = YOLO(str(config.HUB_YOLO_WEIGHTS))
            self.classes = None          # every class in the custom model is "the hub"
            self.source = "custom hub model"
        else:
            self.model = YOLO(config.COCO_YOLO_WEIGHTS)   # downloads ~6 MB on first run
            self.classes = config.COCO_PADDLE_CLASSES
            self.source = "pretrained COCO (" + ", ".join(sorted(self.classes)) + ")"
        print(f"YOLO paddle detector: {self.source}")

    def detect(self, frame_bgr):
        """All paddle-like boxes as [(x1, y1, x2, y2, conf, name)], normalized 0..1."""
        h, w = frame_bgr.shape[:2]
        result = self.model.predict(frame_bgr, imgsz=320, conf=config.YOLO_MIN_CONF, verbose=False)[0]
        boxes = []
        for box in result.boxes:
            name = result.names[int(box.cls)]
            if self.classes is not None and name not in self.classes:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append((x1 / w, y1 / h, x2 / w, y2 / h, float(box.conf), name))
        return boxes


class TagReader:
    """Reports a tag id only once it's been seen TAG_HOLD_FRAMES frames in a
    row, so a flicker of a misread tag can't change the level."""

    def __init__(self):
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self._detector = cv2.aruco.ArucoDetector(dictionary, params)
        self._candidate = None
        self._count = 0
        self.visible = []   # [(id, corners)] this frame, for drawing

    def process(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        self.visible = [] if ids is None else list(zip(ids.flatten().tolist(), corners))
        if not self.visible:
            self._candidate, self._count = None, 0
            return None
        # biggest tag wins if several are in view
        tag_id, _ = max(self.visible, key=lambda v: cv2.contourArea(v[1][0]))
        if tag_id == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = tag_id, 1
        if self._count == config.TAG_HOLD_FRAMES:   # fire once per hold
            return tag_id
        return None


class PaddleLocator:
    """Combines YOLO (where the hub is) with pose (where your wrist is).

    - Hub box near your wrist -> use the box: that's the paddle you're holding.
    - Hub box but no wrist     -> use the box.
    - Wrist but no hub box     -> use the wrist (the hub is in that hand).
    - Neither                  -> keep the last position.
    The result is mapped from CAMERA_X_RANGE to 0..1 across the table,
    mirrored, and smoothed.
    """

    def __init__(self):
        self.x = 0.5
        self.source = "none"
        self.box = None

    def update(self, wrist, boxes):
        raw = None
        self.box = None
        if boxes:
            if wrist is not None:
                def dist(b):
                    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    return (cx - wrist[0]) ** 2 + (cy - wrist[1]) ** 2
                best = min(boxes, key=dist)
                if dist(best) < 0.25 ** 2:
                    self.box = best
            else:
                self.box = max(boxes, key=lambda b: b[4])
        if self.box is not None:
            raw = (self.box[0] + self.box[2]) / 2
            self.source = "yolo"
        elif wrist is not None:
            raw = wrist[0]
            self.source = "pose"
        else:
            self.source = "none"
            return self.x
        self.x = self.smooth(camera_x_to_table(raw))
        return self.x

    def set_direct(self, table_x):
        """No-camera mode: the mouse is the paddle."""
        self.source = "mouse"
        self.x = float(np.clip(table_x, 0, 1))
        return self.x

    def smooth(self, target):
        a = config.PADDLE_SMOOTHING
        return a * self.x + (1 - a) * target


def camera_x_to_table(x_cam):
    """Un-mirrored camera x (0..1) -> table x (0..1), mirrored so moving
    your hand to your right moves the paddle right on screen."""
    lo, hi = config.CAMERA_X_RANGE
    mirrored = 1.0 - x_cam
    return float(np.clip((mirrored - (1 - hi)) / (hi - lo), 0.0, 1.0))


class VisionWorker:
    """Runs the camera and all three vision models on a background thread.

    The game loop never waits on the camera, pose or YOLO: it just asks for
    the most recent results with latest(). If a model is slow or the webcam
    hiccups, the camera view lags a moment but the game, the ball and the
    window keep running. Any step that takes over SLOW_MS gets printed so
    remaining slowdowns can be traced.
    """

    SLOW_MS = 250

    def __init__(self, camera_index, use_yolo=True):
        import threading

        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always the newest frame, not a backlog
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open camera {camera_index}. Try --camera 1, or --no-camera.")

        self.pose = PoseTracker()
        self.tags = TagReader()
        self.hub = HubDetector() if use_yolo else None
        if self.hub is not None:
            # YOLO's first prediction takes several seconds -- do it now,
            # not in the middle of a rally
            print("Warming up YOLO...")
            self.hub.detect(np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), np.uint8))

        self.want_tags = True     # the game turns this off during rallies
        self._lock = threading.Lock()
        self._snapshot = dict(seq=0, frame=None, wrist=None, boxes=[], landmarks=None, tags=[])
        self._tag_events = []
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        import time

        t0 = time.monotonic()
        frame_i = 0
        boxes = []
        bad_reads = 0
        while self._running:
            start = time.monotonic()
            ok, frame = self.cap.read()
            if not ok:
                bad_reads += 1
                if bad_reads % 30 == 1:
                    print(f"Camera returned no frame ({bad_reads} in a row)")
                time.sleep(0.01)
                continue
            bad_reads = 0
            frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
            t_read = time.monotonic()

            wrist = self.pose.process(frame, int((t_read - t0) * 1000))
            t_pose = time.monotonic()

            if self.hub is not None and frame_i % config.YOLO_EVERY_N_FRAMES == 0:
                boxes = self.hub.detect(frame)
            t_yolo = time.monotonic()

            tag = self.tags.process(frame) if self.want_tags else None
            if not self.want_tags:
                self.tags.visible = []
            t_tags = time.monotonic()

            with self._lock:
                self._snapshot = dict(
                    seq=self._snapshot["seq"] + 1, frame=frame, wrist=wrist, boxes=boxes,
                    landmarks=self.pose.landmarks, tags=list(self.tags.visible))
                if tag is not None:
                    self._tag_events.append(tag)
            frame_i += 1

            total_ms = (t_tags - start) * 1000
            if total_ms > self.SLOW_MS:
                print(f"[slow camera frame] {total_ms:.0f} ms = read {(t_read - start) * 1000:.0f}"
                      f" + pose {(t_pose - t_read) * 1000:.0f} + yolo {(t_yolo - t_pose) * 1000:.0f}"
                      f" + tags {(t_tags - t_yolo) * 1000:.0f}")

    def latest(self):
        with self._lock:
            return dict(self._snapshot)

    def pop_tag(self):
        """The most recent tag that was held up long enough, or None."""
        with self._lock:
            tag = self._tag_events[-1] if self._tag_events else None
            self._tag_events.clear()
        return tag

    def close(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()
        self.pose.close()
