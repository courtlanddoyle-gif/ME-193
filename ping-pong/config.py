"""
Every tunable number for the game lives here, so you can change the feel
of the game without digging through the code.
"""

from pathlib import Path

HERE = Path(__file__).parent
MODELS_DIR = HERE / "models"
DATA_DIR = HERE / "data"

# --------------------------------------------------------------------------
# Live score (MQTT)
# --------------------------------------------------------------------------
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
SCORE_TOPIC = "ME193/Rogers/Courtland"
BEST_SCORE_FILE = DATA_DIR / "best_score.json"   # so the record survives restarts

# --------------------------------------------------------------------------
# LEGO Double Motor (the paddle)
# --------------------------------------------------------------------------
HUB_CARD_SERIAL = None   # None = connect to the first Double Motor found
HUB_CARD_COLOR = None
IMU_NOTIFY_MS = 20       # ask the hub for IMU data every 20 ms...
IMU_RATE_HZ = 22         # ...but hub_check.py measured ~22 Hz actually arriving

# --------------------------------------------------------------------------
# Swing detection (IMU classifier)
# --------------------------------------------------------------------------
SWING_MODEL_PATH = MODELS_DIR / "swing_classifier.joblib"
SWING_WINDOW_S = 0.30       # each prediction looks at the last 0.3 s of IMU data
SWING_THRESHOLD = 0.60      # classifier probability needed to call it a swing
SWING_LATCH_S = 0.25        # a swing still "counts" this long after it's detected
SWING_COOLDOWN_S = 0.45     # one physical swing can only hit one ball

# --------------------------------------------------------------------------
# Camera + vision
# --------------------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
PADDLE_HAND = "right"       # which wrist pose tracking follows
# Only the middle of the camera image maps onto the table, so you don't
# have to walk out of frame to reach the edges.
CAMERA_X_RANGE = (0.20, 0.80)
PADDLE_SMOOTHING = 0.5      # 0 = no smoothing, closer to 1 = smoother but laggier

# YOLO: custom weights trained on photos of the hub, if they exist.
# Otherwise fall back to the pretrained COCO model and look for objects that
# the hub (or a ping pong paddle) tends to be detected as.
HUB_YOLO_WEIGHTS = MODELS_DIR / "hub_yolo.pt"
COCO_YOLO_WEIGHTS = "yolov8n.pt"
COCO_PADDLE_CLASSES = {"tennis racket", "cell phone", "remote", "mouse"}
YOLO_EVERY_N_FRAMES = 3     # YOLO is the slowest step; don't run it every frame
YOLO_MIN_CONF = 0.35

# --------------------------------------------------------------------------
# AprilTags (tag36h11 family; print them with make_tags.py)
# --------------------------------------------------------------------------
START_TAG_ID = 0
LEVEL_TAG_IDS = {1: "easy", 2: "medium", 3: "hard"}
TAG_HOLD_FRAMES = 5         # tag must be seen this many frames in a row to count

# --------------------------------------------------------------------------
# Whistle serve
# --------------------------------------------------------------------------
MIC_SAMPLE_RATE = 44100
MIC_BLOCK = 2048                  # ~46 ms of audio per FFT
WHISTLE_BAND_HZ = (600, 4000)     # where whistles live
WHISTLE_TONALITY = 25.0           # peak power / median power in the band
WHISTLE_MIN_RMS = 0.01            # ignore near-silence
WHISTLE_MIN_BLOCKS = 6            # ~0.3 s of steady whistle to serve

# --------------------------------------------------------------------------
# Levels: ball speed (table lengths per second), paddle size, and the
# opponent's PD gains and how often it aims away from you.
# --------------------------------------------------------------------------
LEVELS = {
    "easy":   dict(ball_speed=0.45, paddle_width=0.30, kp=18.0, kd=6.0, aim_away=0.2),
    "medium": dict(ball_speed=0.70, paddle_width=0.22, kp=30.0, kd=8.0, aim_away=0.5),
    "hard":   dict(ball_speed=1.00, paddle_width=0.16, kp=60.0, kd=12.0, aim_away=0.8),
}
DEFAULT_LEVEL = "easy"

# The ball can only be hit while it's in the last stretch of your side.
CONTACT_ZONE = (0.82, 1.0)        # fraction of table length (0 = opponent, 1 = you)
MISS_PAUSE_S = 1.5
