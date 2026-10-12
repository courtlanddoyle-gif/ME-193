"""
Virtual ping pong with the LEGO Double Motor as your paddle.

    python game.py                  # the real thing: hub + camera + mic
    python game.py --sim            # no hub: SPACE = swing
    python game.py --sim --no-camera --no-mic   # keyboard + mouse only

Keyboard shortcuts (always work, handy for testing):
    1 / 2 / 3   act like level tags (easy / medium / hard)
    0           acts like the start tag
    w           acts like a whistle (serve)
    SPACE       swing (sim paddle only)
    n           new game (restart at 0-0, same level)
    r           reset the record to 0
    q / ESC     quit
"""

import argparse
import time

import cv2
import numpy as np

import config
from game_logic import GAME_OVER, OPPONENT_WIDTH, RALLY, Game
from imu_swing import SwingDetector
from paddle_hw import make_paddle
from score import ScorePublisher, load_best, save_best

CAM_W, CAM_H = config.FRAME_WIDTH, config.FRAME_HEIGHT
TABLE_W = 420
HUD_H = 90
WINDOW = "Ping Pong (q to quit)"

# table drawing area inside the table panel
TX0, TY0, TX1, TY1 = 50, 30, TABLE_W - 50, CAM_H - 30

# What each rumble means, shown on screen while the hub is buzzing
HAPTIC_LABELS = {
    "serve": "HUB: 1 kick = serve",
    "hit": "HUB: 2 kicks = hit!",
    "record": "HUB: 3 pulses = NEW RECORD!",
    "miss": "HUB: long buzz = miss",
    "win": "HUB: victory dance!",
    "lose": "HUB: sad buzz = game over",
}
# If several things happen in one tick (e.g. a winning point is also a
# game win), only the most important rumble plays.
HAPTIC_PRIORITY = ["game_won", "game_lost", "record", "miss", "hit", "serve"]
HAPTIC_FOR_EVENT = {"game_won": "win", "game_lost": "lose"}
HAPTIC_LABEL_S = 1.2

SLOW_LOOP_S = 0.15     # print a breakdown of any game-loop tick slower than this
KEY_TAGS = {ord("0"): 0, ord("1"): 1, ord("2"): 2, ord("3"): 3}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim", action="store_true", help="simulated paddle (no LEGO hub)")
    p.add_argument("--no-camera", action="store_true", help="mouse controls the paddle, keys act as tags")
    p.add_argument("--no-yolo", action="store_true", help="track the paddle with pose only")
    p.add_argument("--no-mic", action="store_true", help="press w instead of whistling")
    p.add_argument("--no-mqtt", action="store_true", help="don't post the score")
    p.add_argument("--broker", default=config.MQTT_BROKER)
    p.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    return p.parse_args()


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

def tx(x):
    return int(TX0 + x * (TX1 - TX0))


def ty(y):
    return int(TY0 + y * (TY1 - TY0))


def draw_hub_paddle(img, cx, cy, width_px, color, thick=2):
    """The on-screen paddle is the outline of the Double Motor: a rounded
    body with the two motor hubs on its sides."""
    w, h = width_px, 22
    x0, y0, x1, y1 = cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2
    r = 8
    cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, thick)
    cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, thick)
    for px in (x0 + r, x1 - r):
        for py in (y0 + r, y1 - r):
            cv2.circle(img, (px, py), r, color, thick)
    cv2.rectangle(img, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), (30, 30, 30), -1)
    for px in (x0 + h // 2, x1 - h // 2):           # the two motor outputs
        cv2.circle(img, (px, cy), 6, color, thick)
        cv2.line(img, (px - 3, cy), (px + 3, cy), color, 1)
        cv2.line(img, (px, cy - 3), (px, cy + 3), color, 1)
    cv2.circle(img, (cx, cy), 3, color, -1)          # the button light


class Celebration:
    """Confetti bursts over the whole window when you win a game."""

    COLORS = [(60, 60, 255), (60, 220, 255), (80, 255, 120), (255, 160, 60),
              (255, 80, 200), (255, 255, 255)]

    def __init__(self):
        self.particles = []
        self._bursts = []
        self._last = 0.0

    def start(self, now):
        self._bursts = [now, now + 1.2, now + 2.4]   # three waves of confetti
        self._last = now

    def _burst(self, width):
        rng = np.random.default_rng()
        for _ in range(140):
            self.particles.append([
                rng.uniform(0, width), rng.uniform(-60, 0),      # x, y
                rng.uniform(-60, 60), rng.uniform(80, 260),      # vx, vy (px/s)
                self.COLORS[rng.integers(len(self.COLORS))],
                int(rng.integers(3, 7)),                        # size
            ])

    def draw(self, screen, now):
        dt = min(now - self._last, 0.05)
        self._last = now
        h, w = screen.shape[:2]
        while self._bursts and now >= self._bursts[0]:
            self._bursts.pop(0)
            self._burst(w)
        for p in self.particles:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[3] += 120 * dt                                     # gravity
            cv2.rectangle(screen, (int(p[0]), int(p[1])),
                          (int(p[0]) + p[5], int(p[1]) + p[5]), p[4], -1)
        self.particles = [p for p in self.particles if p[1] < h]

    @property
    def active(self):
        return bool(self.particles or self._bursts)


def draw_banner(screen, now, game):
    """Big end-of-game banner across the middle of the window."""
    h, w = screen.shape[:2]
    won = game.winner == "player"
    title = "YOU WIN!" if won else "CPU WINS"
    score = f"{game.player_points} - {game.opponent_points}"
    pulse = 1.0 + (0.08 * np.sin(now * 6) if won else 0)
    overlay = screen.copy()
    cv2.rectangle(overlay, (0, h // 2 - 90), (w, h // 2 + 80), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, screen, 0.3, 0, dst=screen)
    color = (80, 255, 120) if won else (160, 160, 160)
    for text, scale, y in ((title, 2.6 * pulse, h // 2 - 10), (score, 1.3, h // 2 + 40)):
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 4)
        cv2.putText(screen, text, ((w - tw) // 2, y), cv2.FONT_HERSHEY_DUPLEX, scale, color, 4)
    hint = "Show tag 0 or press n for a new game"
    (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(screen, hint, ((w - tw) // 2, h // 2 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (220, 220, 220), 1)


def draw_table(game, paddle_x, swinging):
    img = np.full((CAM_H, TABLE_W, 3), (40, 28, 18), np.uint8)
    cv2.rectangle(img, (TX0, TY0), (TX1, TY1), (110, 70, 20), -1)
    cv2.rectangle(img, (TX0, TY0), (TX1, TY1), (240, 240, 240), 2)
    cv2.line(img, ((TX0 + TX1) // 2, TY0), ((TX0 + TX1) // 2, TY1), (200, 200, 200), 1)
    cv2.line(img, (TX0 - 10, ty(0.5)), (TX1 + 10, ty(0.5)), (255, 255, 255), 3)   # net

    zone = img.copy()
    cv2.rectangle(zone, (TX0, ty(config.CONTACT_ZONE[0])), (TX1, TY1), (60, 180, 60), -1)
    img = cv2.addWeighted(zone, 0.25, img, 0.75, 0)

    opp = game.opponent
    draw_hub_paddle(img, tx(opp.x), TY0 - 4, int(OPPONENT_WIDTH * (TX1 - TX0)), (80, 80, 240))

    if game.state == RALLY:
        b = game.ball
        r = int(6 + 7 * b.height)
        cv2.circle(img, (tx(b.x) + 4, ty(b.y) + 4), 6, (20, 20, 20), -1)   # shadow
        cv2.circle(img, (tx(b.x), ty(b.y) - int(10 * b.height)), r, (240, 245, 255), -1)
        if b.toward_player:   # show where it's headed so you can get there
            cv2.drawMarker(img, (tx(b.landing_x), TY1), (0, 220, 255), cv2.MARKER_TRIANGLE_UP, 14, 2)

    # scoreboard
    board = f"YOU {game.player_points}   CPU {game.opponent_points}"
    (bw, _), _ = cv2.getTextSize(board, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (TX0 + 6, ty(0.5) + 10), (TX0 + 22 + bw, ty(0.5) + 40), (20, 20, 20), -1)
    cv2.putText(img, board, (TX0 + 14, ty(0.5) + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, f"first to {config.POINTS_TO_WIN}", (TX0 + 14, ty(0.5) + 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    pw = int(game.params["paddle_width"] * (TX1 - TX0))
    color = (80, 255, 120) if swinging else (230, 230, 230)
    draw_hub_paddle(img, tx(paddle_x), TY1 + 4, pw, color, 3 if swinging else 2)
    return img


def draw_camera(frame, landmarks, locator, visible_tags, wrist, sources_note):
    """Mirror the frame and draw what each model sees on top of it."""
    view = cv2.flip(frame, 1)
    def px(x, y):
        return int((1 - x) * CAM_W), int(y * CAM_H)

    lo, hi = config.CAMERA_X_RANGE
    for edge in (lo, hi):
        cv2.line(view, (int(edge * CAM_W), 0), (int(edge * CAM_W), CAM_H), (90, 90, 90), 1)

    if landmarks is not None:
        lm = landmarks
        for a, b in [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]:
            cv2.line(view, px(lm[a].x, lm[a].y), px(lm[b].x, lm[b].y), (255, 200, 0), 2)
    if wrist is not None:
        cv2.circle(view, px(*wrist), 10, (0, 255, 255), 2)
    if locator.box is not None:
        x1, y1, x2, y2, conf, name = locator.box
        cv2.rectangle(view, px(x2, y1), px(x1, y2), (0, 255, 0), 2)
        cv2.putText(view, f"paddle ({name} {conf:.2f})", px(x2, y1 - 0.01),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if visible_tags:
        for tag_id, corners in visible_tags:
            pts = np.array([px(c[0] / CAM_W, c[1] / CAM_H) for c in corners[0]], np.int32)
            cv2.polylines(view, [pts], True, (255, 0, 255), 3)
            label = {config.START_TAG_ID: "START"}.get(tag_id, config.LEVEL_TAG_IDS.get(tag_id, "?"))
            cv2.putText(view, f"tag {tag_id}: {label}", tuple(pts[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(view, sources_note, (10, CAM_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return view


def no_camera_panel():
    img = np.full((CAM_H, CAM_W, 3), 25, np.uint8)
    lines = ["NO-CAMERA MODE", "", "Move the mouse over the table = paddle",
             "1/2/3 = level tags, 0 = start tag", "w = whistle (if --no-mic)",
             "SPACE = swing (sim paddle)"]
    for i, line in enumerate(lines):
        cv2.putText(img, line, (30, 80 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)
    return img


def bar(img, x, y, label, value, threshold=None, width=150):
    cv2.putText(img, label, (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    bx = x + 70
    cv2.rectangle(img, (bx, y), (bx + width, y + 14), (70, 70, 70), 1)
    v = float(np.clip(value, 0, 1))
    color = (80, 255, 120) if threshold is not None and v >= threshold else (0, 180, 255)
    cv2.rectangle(img, (bx + 1, y + 1), (bx + 1 + int(v * (width - 2)), y + 13), color, -1)
    if threshold is not None:
        tx_ = bx + int(threshold * width)
        cv2.line(img, (tx_, y - 2), (tx_, y + 16), (255, 255, 255), 1)


def draw_hud(game, swing_prob, mic_level, locator_source, mqtt_status, total_w, haptic_label=None):
    img = np.full((HUD_H, total_w, 3), 15, np.uint8)
    big = (f"{game.score_text}    STREAK {game.streak}    RECORD {game.best}    "
           f"LEVEL {game.level.upper()}    [{game.state}]")
    cv2.putText(img, big, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(img, game.message, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)
    bar(img, 12, 66, "swing P", swing_prob, config.SWING_THRESHOLD)
    if mic_level is not None:
        bar(img, 250, 66, "whistle", mic_level[2] / (2 * config.WHISTLE_TONALITY), 0.5)
    cv2.putText(img, f"paddle: {locator_source}   mqtt: {mqtt_status}", (500, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)
    if haptic_label:
        cv2.putText(img, haptic_label, (total_w - 330, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 200, 255), 2)
    return img


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def main():
    args = parse_args()

    from vision import PaddleLocator
    locator = PaddleLocator()
    detector = SwingDetector()
    paddle = make_paddle(args.sim)

    vision = None
    if not args.no_camera:
        from vision import VisionWorker
        vision = VisionWorker(args.camera, use_yolo=not args.no_yolo)

    mic = None
    if not args.no_mic:
        from whistle import WhistleListener
        try:
            mic = WhistleListener()
        except Exception as exc:
            print(f"Microphone unavailable ({exc}); press w to serve instead.")

    game = Game(best=load_best())
    publisher = ScorePublisher(enabled=not args.no_mqtt, broker=args.broker)
    publisher.publish(game.best)

    mouse = {"x": 0.5}
    def on_mouse(event, x, y, flags, param):
        mouse["x"] = (x - CAM_W - TX0) / (TX1 - TX0)
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    seen_seq = 0
    snap = None
    record_rumbled = False   # the record rumble plays once per rally, not on every hit past it
    celebration = Celebration()
    paddle_x = 0.5
    last = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            dt = min(now - last, 0.05)
            last = now
            tag = None

            # ---- perception ----------------------------------------------
            # The camera + models run on their own thread (VisionWorker);
            # here we only pick up whatever they finished most recently.
            if vision is not None:
                vision.want_tags = game.state != RALLY   # tags work between rallies and after a game
                snap = vision.latest()
                if snap["seq"] != seen_seq:          # a new camera frame is ready
                    seen_seq = snap["seq"]
                    paddle_x = locator.update(snap["wrist"], snap["boxes"])
                tag = vision.pop_tag()
            else:
                paddle_x = locator.set_direct(mouse["x"])
            t_vision = time.monotonic()

            detector.update(now, paddle.imu_window(now))
            swinging = detector.swinging(now)
            whistle = mic.poll() if mic is not None else False

            # ---- keyboard ------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in KEY_TAGS:
                tag = KEY_TAGS[key]
            if key == ord("w"):
                whistle = True
            restart = key == ord("n")
            if key == ord(" "):
                paddle.swing()
            if key == ord("r"):
                game.best = 0
                save_best(0)
                publisher.publish(0)

            # ---- policy --------------------------------------------------
            t_input = time.monotonic()
            events = game.update(now, dt, paddle_x, swinging, tag, whistle, restart)

            # ---- actions -------------------------------------------------
            if "hit" in events:
                detector.consume(now)
            if "serve" in events or "miss" in events:
                record_rumbled = False
            if "record" in events:
                save_best(game.best)
                publisher.publish(game.best)
            if "game_won" in events:
                celebration.start(now)
            if "new_game" in events:
                celebration = Celebration()
            rumble_events = [e for e in events if not (e == "record" and record_rumbled)]
            for e in HAPTIC_PRIORITY:
                if e in rumble_events:
                    paddle.haptic(HAPTIC_FOR_EVENT.get(e, e))
                    record_rumbled = record_rumbled or e == "record"
                    break
            for e in events:
                if e != "hit":
                    print(f"[{game.state}] {e}: {game.message}")

            # ---- draw ----------------------------------------------------
            t_actions = time.monotonic()
            h_event, h_time = paddle.last_haptic
            haptic_label = HAPTIC_LABELS.get(h_event) if now - h_time < HAPTIC_LABEL_S else None
            table = draw_table(game, paddle_x, swinging)
            if snap is not None and snap["frame"] is not None:
                note = f"{'SIM paddle' if paddle.is_sim else 'LEGO paddle'} | paddle from {locator.source}"
                left = draw_camera(snap["frame"], snap["landmarks"], locator, snap["tags"],
                                   snap["wrist"], note)
            else:
                left = no_camera_panel()
            screen = np.vstack([np.hstack([left, table]),
                                draw_hud(game, detector.prob, mic.level if mic else None,
                                         locator.source, publisher.status, CAM_W + TABLE_W,
                                         haptic_label)])
            if game.state == GAME_OVER:
                draw_banner(screen, now, game)
            if celebration.active:
                celebration.draw(screen, now)
            dx, dy = paddle.shake_offset(now)
            if dx or dy:
                screen = np.roll(screen, (dy, dx), axis=(0, 1))
            cv2.imshow(WINDOW, screen)

            # ---- freeze detective ----------------------------------------
            end = time.monotonic()
            if end - now > SLOW_LOOP_S:
                print(f"[slow game loop] {(end - now) * 1000:.0f} ms = vision {(t_vision - now) * 1000:.0f}"
                      f" + swing/mic/keys {(t_input - t_vision) * 1000:.0f}"
                      f" + game/haptics/mqtt {(t_actions - t_input) * 1000:.0f}"
                      f" + draw {(end - t_actions) * 1000:.0f}")
    finally:
        paddle.close()
        if vision is not None:
            vision.close()
        if mic is not None:
            mic.close()
        publisher.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
