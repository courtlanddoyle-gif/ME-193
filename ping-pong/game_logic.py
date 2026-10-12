"""
The rules of the game, with no camera, mic, motor or window involved, so
every rule can be unit tested (see tests/).

Table coordinates: x runs 0 (left) -> 1 (right); y runs 0 (opponent's end)
-> 1 (your end).

States:
    SETUP      -- show a level tag (1/2/3), then the start tag (0)
    READY      -- armed; whistle to serve (level tags still work here)
    RALLY      -- ball in play
    POINT      -- short pause after a point, then back to READY
    GAME_OVER  -- someone reached POINTS_TO_WIN; tag 0 (or 'n') for a new game

Scoring, like real ping pong but to 5: you win the point if the opponent
can't reach your return; the opponent wins it if you miss. Separately, the
streak counts your continuous hits and the record is the longest streak
ever -- that's what gets posted over MQTT.

The policy -- how the game decides what happens each tick -- is the
update() method. It's a rule-based state machine fed by the learned
perception models (pose/YOLO for paddle position, the IMU classifier for
swinging), plus a PD controller for the opponent.
"""

import random
from dataclasses import dataclass, field

import numpy as np

import config

SETUP, READY, RALLY, POINT, GAME_OVER = "SETUP", "READY", "RALLY", "POINT", "GAME_OVER"
OPPONENT_WIDTH = 0.22
SPEEDUP_PER_HIT = 0.03      # each hit in a rally makes the ball 3% faster...
MAX_SPEEDUP = 1.5           # ...up to 1.5x the level's base speed


class PDPaddle:
    """The opponent's paddle, driven by a PD controller toward where it
    predicts the ball will land:  accel = Kp * error - Kd * velocity.
    Higher Kp = reacts harder; higher Kd = less overshoot/wobble.
    Its speed is capped at vmax, like a real player's (or motor's) top
    speed -- without the cap it could teleport and would never miss."""

    def __init__(self, x=0.5):
        self.x = x
        self.v = 0.0
        self.target = x

    def update(self, dt, kp, kd, vmax=float("inf")):
        error = self.target - self.x
        accel = kp * error - kd * self.v
        self.v = float(np.clip(self.v + accel * dt, -vmax, vmax))
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
    last_power: float = 0.0       # power of your most recent hit, 0..1
    player_points: int = 0
    opponent_points: int = 0
    winner: str = None            # "player" or "opponent" once the game is over
    _pause_until: float = 0.0
    _react_at: float = 0.0        # opponent starts chasing your return at this time
    rng: random.Random = field(default_factory=random.Random)

    @property
    def params(self):
        return config.LEVELS[self.level]

    @property
    def ball_speed(self):
        boost = min(MAX_SPEEDUP, 1 + SPEEDUP_PER_HIT * self.rally_hits)
        return self.params["ball_speed"] * boost

    # -- the policy ---------------------------------------------------------

    @property
    def score_text(self):
        return f"YOU {self.player_points} - {self.opponent_points} CPU"

    def update(self, now, dt, paddle_x, swinging, tag=None, whistle=False, restart=False,
               swing_power=0.5):
        """Advance the game one tick. Returns a list of event names that
        happened ("level", "armed", "new_game", "serve", "hit", "smash",
        "record", "miss", "winner", "game_won", "game_lost") so the caller
        can trigger haptics, MQTT, sounds.

        swing_power (0..1, from the IMU) sets how fast your return goes."""
        events = []

        if restart or (tag == config.START_TAG_ID and self.state == GAME_OVER):
            self.new_game()
            return ["new_game"]

        # Tags only matter between rallies.
        if tag is not None and self.state in (SETUP, READY, GAME_OVER):
            if tag in config.LEVEL_TAG_IDS:
                self.level = config.LEVEL_TAG_IDS[tag]
                next_step = "whistle to serve!" if self.state == READY else "show tag 0 to start"
                self.message = f"Level: {self.level.upper()}  --  {next_step}"
                events.append("level")
            elif tag == config.START_TAG_ID and self.state == SETUP:
                self.state = READY
                self.message = f"First to {config.POINTS_TO_WIN}! Whistle to serve."
                events.append("armed")

        if self.state == READY and whistle:
            self._serve()
            events.append("serve")

        if self.state == POINT and now >= self._pause_until:
            self.state = READY
            self.message = f"{self.score_text}  --  whistle to serve!"

        if self.state == RALLY:
            events += self._rally_tick(now, dt, paddle_x, swinging, swing_power)

        p = self.params
        self.opponent.update(dt, p["kp"], p["kd"], p["opp_vmax"])
        return events

    def new_game(self):
        """0-0, same level, ready to serve. The record survives; the streak
        starts fresh."""
        self.player_points = self.opponent_points = 0
        self.winner = None
        self.streak = 0
        self.rally_hits = 0
        self.state = READY
        self.opponent.x = self.opponent.target = 0.5
        self.opponent.v = 0.0
        self.message = f"New game, first to {config.POINTS_TO_WIN}! Whistle to serve."

    def _rally_tick(self, now, dt, paddle_x, swinging, swing_power=0.5):
        ball = self.ball
        ball.step(dt)
        half = self.params["paddle_width"] / 2

        if ball.toward_player:
            self.opponent.target = 0.5      # opponent drifts back to center
            in_zone = config.CONTACT_ZONE[0] <= ball.y <= config.CONTACT_ZONE[1]
            lined_up = abs(ball.x - paddle_x) <= half
            if in_zone and lined_up and swinging:
                return self._player_hit(now, paddle_x, swing_power)
            if ball.y > config.CONTACT_ZONE[1]:
                return self._player_miss(now, lined_up)
        else:
            # the opponent needs a moment to read your shot before chasing it
            if now >= self._react_at:
                self.opponent.target = ball.landing_x
            if ball.y <= 0.0:
                if abs(self.opponent.x - ball.x) <= OPPONENT_WIDTH / 2:
                    self._opponent_return(paddle_x)
                else:
                    # PD controller couldn't get there in time: your point,
                    # and your streak survives
                    self.player_points += 1
                    self.message = "Winner! Opponent couldn't reach it."
                    return ["winner"] + self._end_point(now)
        return []

    def _player_hit(self, now, paddle_x, power=0.5):
        events = ["hit"]
        self.last_power = power
        self._react_at = now + self.params["opp_delay"]
        self.streak += 1
        self.rally_hits += 1
        if self.streak > self.best:
            self.best = self.streak
            events.append("record")
        smash = power >= config.SMASH_POWER
        if smash:
            events.append("smash")
        self.message = f"{'SMASH!' if smash else 'Hit!'} Streak {self.streak}  ({power:.0%} power)"
        # Like Pong: where the ball meets the paddle sets the return angle.
        offset = (self.ball.x - paddle_x) / (self.params["paddle_width"] / 2)
        to_x = float(np.clip(self.ball.x + 0.35 * offset, 0.05, 0.95))
        # Swing power: a harder swing sends a faster return, which gives the
        # opponent less time to get there.
        slow, fast = config.RETURN_SPEED_RANGE
        self.ball.launch(self.ball.x, self.ball.y, to_x, 0.0,
                         self.ball_speed * (slow + (fast - slow) * power))
        return events

    def _player_miss(self, now, lined_up):
        self.message = ("Miss -- you were there but didn't swing!" if lined_up
                        else "Miss -- paddle wasn't in the right spot!")
        self.streak = 0
        self.opponent_points += 1
        return ["miss"] + self._end_point(now)

    def _end_point(self, now):
        """After any point: game over at POINTS_TO_WIN, otherwise a short pause."""
        self.rally_hits = 0
        if max(self.player_points, self.opponent_points) >= config.POINTS_TO_WIN:
            self.state = GAME_OVER
            won = self.player_points > self.opponent_points
            self.winner = "player" if won else "opponent"
            self.message = (f"YOU WIN {self.player_points}-{self.opponent_points}!" if won
                            else f"CPU wins {self.opponent_points}-{self.player_points}.")
            self.message += "  Tag 0 or 'n' for a new game."
            return ["game_won" if won else "game_lost"]
        self.state = POINT
        self._pause_until = now + config.POINT_PAUSE_S
        return []

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
        self.message = f"Rally!  {self.score_text}"
