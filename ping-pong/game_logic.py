"""
The rules of the game, with no camera, mic, motor or window involved, so
every rule can be unit tested (see tests/).

Table coordinates: x runs 0 (left) -> 1 (right); y runs 0 (opponent's end)
-> 1 (your end).

States:
    SETUP  -- show a level tag (1/2/3), then the start tag (0)
    READY  -- armed; whistle to serve (level tags still work here)
    RALLY  -- ball in play
    MISS   -- short pause after you miss, then back to READY

The policy -- how the game decides what happens each tick -- is the
update() method. It's a rule-based state machine fed by the learned
perception models (pose/YOLO for paddle position, the IMU classifier for
swinging), plus a PD controller for the opponent.
"""

import random
from dataclasses import dataclass, field

import numpy as np

import config

SETUP, READY, RALLY, MISS = "SETUP", "READY", "RALLY", "MISS"
OPPONENT_WIDTH = 0.22
SPEEDUP_PER_HIT = 0.03      # each hit in a rally makes the ball 3% faster...
MAX_SPEEDUP = 1.5           # ...up to 1.5x the level's base speed


class PDPaddle:
    """The opponent's paddle, driven by a PD controller toward where it
    predicts the ball will land:  accel = Kp * error - Kd * velocity.
    Higher Kp = reacts harder; higher Kd = less overshoot/wobble."""

    def __init__(self, x=0.5):
        self.x = x
        self.v = 0.0
        self.target = x

    def update(self, dt, kp, kd):
        error = self.target - self.x
        accel = kp * error - kd * self.v
        self.v += accel * dt
        self.x += self.v * dt
        if not 0.0 <= self.x <= 1.0:          # walls
            self.x = float(np.clip(self.x, 0.0, 1.0))
            self.v = 0.0


@dataclass
class Ball:
    x: float = 0.5
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    y0: float = 0.0
    y1: float = 1.0

    def launch(self, from_x, from_y, to_x, to_y, speed):
        """Fly from one end to the other at `speed` table-lengths/second."""
        flight_time = abs(to_y - from_y) / speed
        self.x, self.y = from_x, from_y
        self.y0, self.y1 = from_y, to_y
        self.vx = (to_x - from_x) / flight_time
        self.vy = speed if to_y > from_y else -speed

    def step(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt

    @property
    def toward_player(self):
        return self.vy > 0

    @property
    def landing_x(self):
        """Where the ball will cross the far end, if nothing touches it."""
        t = (self.y1 - self.y) / self.vy if self.vy else 0.0
        return self.x + self.vx * t

    @property
    def height(self):
        """0 at each end, 1 mid-table -- purely for drawing an arc."""
        span = abs(self.y1 - self.y0) or 1.0
        p = float(np.clip(abs(self.y - self.y0) / span, 0, 1))
        return 4 * p * (1 - p)


@dataclass
class Game:
    level: str = config.DEFAULT_LEVEL
    state: str = SETUP
    streak: int = 0
    best: int = 0
    message: str = "Show a level tag (1-3), then tag 0 to start"
    ball: Ball = field(default_factory=Ball)
    opponent: PDPaddle = field(default_factory=PDPaddle)
    rally_hits: int = 0
    _pause_until: float = 0.0
    rng: random.Random = field(default_factory=random.Random)

    @property
    def params(self):
        return config.LEVELS[self.level]

    @property
    def ball_speed(self):
        boost = min(MAX_SPEEDUP, 1 + SPEEDUP_PER_HIT * self.rally_hits)
        return self.params["ball_speed"] * boost

    # -- the policy ---------------------------------------------------------

    def update(self, now, dt, paddle_x, swinging, tag=None, whistle=False):
        """Advance the game one tick. Returns a list of event names that
        happened ("level", "armed", "serve", "hit", "miss", "record",
        "winner") so the caller can trigger haptics, MQTT, sounds."""
        events = []

        # Tags only matter between rallies.
        if tag is not None and self.state in (SETUP, READY):
            if tag in config.LEVEL_TAG_IDS:
                self.level = config.LEVEL_TAG_IDS[tag]
                self.message = f"Level: {self.level.upper()}  --  show tag 0 to start"
                if self.state == READY:
                    self.message = f"Level: {self.level.upper()}  --  whistle to serve!"
                events.append("level")
            elif tag == config.START_TAG_ID and self.state == SETUP:
                self.state = READY
                self.message = "Ready -- whistle to serve!"
                events.append("armed")

        if self.state == READY and whistle:
            self._serve()
            events.append("serve")

        if self.state == MISS and now >= self._pause_until:
            self.state = READY
            self.message = "Whistle to serve!"

        if self.state == RALLY:
            events += self._rally_tick(now, dt, paddle_x, swinging)

        self.opponent.update(dt, self.params["kp"], self.params["kd"])
        return events

    def _rally_tick(self, now, dt, paddle_x, swinging):
        ball = self.ball
        ball.step(dt)
        half = self.params["paddle_width"] / 2

        if ball.toward_player:
            self.opponent.target = 0.5      # opponent drifts back to center
            in_zone = config.CONTACT_ZONE[0] <= ball.y <= config.CONTACT_ZONE[1]
            lined_up = abs(ball.x - paddle_x) <= half
            if in_zone and lined_up and swinging:
                return self._player_hit(paddle_x)
            if ball.y > config.CONTACT_ZONE[1]:
                return self._player_miss(now, lined_up)
        else:
            self.opponent.target = ball.landing_x
            if ball.y <= 0.0:
                if abs(self.opponent.x - ball.x) <= OPPONENT_WIDTH / 2:
                    self._opponent_return(paddle_x)
                else:
                    # PD controller didn't get there in time: your point,
                    # streak survives, opponent serves again
                    self.message = "Winner! Opponent couldn't reach it."
                    self._serve()
                    return ["winner"]
        return []

    def _player_hit(self, paddle_x):
        events = ["hit"]
        self.streak += 1
        self.rally_hits += 1
        if self.streak > self.best:
            self.best = self.streak
            events.append("record")
        self.message = f"Hit! Streak {self.streak}"
        # Like Pong: where the ball meets the paddle sets the return angle.
        offset = (self.ball.x - paddle_x) / (self.params["paddle_width"] / 2)
        to_x = float(np.clip(self.ball.x + 0.35 * offset, 0.05, 0.95))
        self.ball.launch(self.ball.x, self.ball.y, to_x, 0.0, self.ball_speed)
        return events

    def _player_miss(self, now, lined_up):
        self.state = MISS
        self._pause_until = now + config.MISS_PAUSE_S
        self.message = ("Miss -- you were there but didn't swing!" if lined_up
                        else "Miss -- paddle wasn't in the right spot!")
        self.streak = 0
        self.rally_hits = 0
        return ["miss"]

    def _opponent_return(self, paddle_x):
        to_x = self._opponent_aim(paddle_x)
        self.ball.launch(self.ball.x, 0.0, to_x, 1.0, self.ball_speed)

    def _opponent_aim(self, paddle_x):
        """The opponent's policy: with probability aim_away, hit to the side
        of the table farthest from your paddle; otherwise anywhere."""
        if self.rng.random() < self.params["aim_away"]:
            far = 0.15 if paddle_x > 0.5 else 0.85
            return float(np.clip(far + self.rng.uniform(-0.07, 0.07), 0.08, 0.92))
        return self.rng.uniform(0.15, 0.85)

    def _serve(self):
        self.state = RALLY
        self.rally_hits = 0
        self.opponent.x = self.opponent.target = 0.5
        self.opponent.v = 0.0
        to_x = self.rng.uniform(0.25, 0.75)    # serves are gentle
        self.ball.launch(0.5, 0.0, to_x, 1.0, self.ball_speed)
        if not self.message.startswith("Winner"):
            self.message = "Rally!"
