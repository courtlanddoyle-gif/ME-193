"""
Tests that run with no hub, camera or microphone:

    python -m pytest tests
"""

import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from game_logic import GAME_OVER, POINT, READY, RALLY, SETUP, Game, PDPaddle  # noqa: E402
from imu_swing import synthetic_window, window_features  # noqa: E402
from score import format_score  # noqa: E402
from vision import camera_x_to_table  # noqa: E402
from whistle import WhistleDetector, analyze_block  # noqa: E402

DT = 1 / 60


def started_game(level="easy"):
    g = Game(rng=random.Random(1))
    g.update(0, DT, 0.5, False, tag={"easy": 1, "medium": 2, "hard": 3}[level])
    g.update(0, DT, 0.5, False, tag=config.START_TAG_ID)
    return g


def play(g, seconds, player, t0=0.0, auto_serve=False):
    """Run the game with a scripted player: player(game) -> (paddle_x, swinging).
    auto_serve whistles whenever the game is waiting for a serve."""
    events = []
    t = t0
    for _ in range(int(seconds / DT)):
        t += DT
        x, swing = player(g)
        events += g.update(t, DT, x, swing, whistle=auto_serve and g.state == READY)
    return events, t


@pytest.fixture
def unbeatable_opponent(monkeypatch):
    """Opponent with no speed limit or delay, so rallies never end on a winner."""
    for level in config.LEVELS:
        monkeypatch.setitem(config.LEVELS[level], "opp_vmax", 100.0)
        monkeypatch.setitem(config.LEVELS[level], "opp_delay", 0.0)


# ---- states and tags --------------------------------------------------------

def test_tags_set_level_and_arm():
    g = Game()
    assert g.state == SETUP
    g.update(0, DT, 0.5, False, tag=3)
    assert g.level == "hard" and g.state == SETUP
    g.update(0, DT, 0.5, False, tag=0)
    assert g.state == READY


def test_whistle_only_serves_when_armed():
    g = Game()
    g.update(0, DT, 0.5, False, whistle=True)
    assert g.state == SETUP
    g = started_game()
    assert "serve" in g.update(0, DT, 0.5, False, whistle=True)
    assert g.state == RALLY


def test_level_tags_ignored_mid_rally():
    g = started_game("easy")
    g.update(0, DT, 0.5, False, whistle=True)
    g.update(0, DT, 0.5, False, tag=3)
    assert g.level == "easy"


# ---- hitting and missing ----------------------------------------------------

def perfect_player(g):
    """Paddle always under the ball, swings once it's in the contact zone."""
    in_zone = g.ball.toward_player and g.ball.y >= config.CONTACT_ZONE[0]
    return g.ball.x, in_zone


def test_perfect_player_builds_a_streak(unbeatable_opponent):
    g = started_game("easy")
    g.update(0, DT, 0.5, False, whistle=True)
    events, _ = play(g, 20, perfect_player)
    assert "miss" not in events
    assert g.streak >= 5 and g.best == g.streak


def test_no_swing_is_a_miss_even_when_lined_up():
    g = started_game()
    g.update(0, DT, 0.5, False, whistle=True)
    t = 0.0
    while g.state == RALLY and t < 5:
        t += DT
        g.update(t, DT, g.ball.x, False)
    assert g.state == POINT
    assert "didn't swing" in g.message
    assert (g.player_points, g.opponent_points) == (0, 1)


def test_wrong_place_is_a_miss_even_when_swinging():
    g = started_game()
    g.update(0, DT, 0.5, False, whistle=True)
    events, _ = play(g, 5, lambda g: ((g.ball.x + 0.5) % 1.0, True))
    assert "miss" in events and g.streak == 0


def test_miss_resets_streak_but_keeps_record_then_ready(unbeatable_opponent):
    g = started_game()
    g.update(0, DT, 0.5, False, whistle=True)
    _, t = play(g, 12, perfect_player)
    best = g.best
    assert best > 0
    events, t = play(g, 5, lambda g: ((g.ball.x + 0.5) % 1.0, False), t0=t)
    assert "miss" in events
    assert g.streak == 0 and g.best == best
    play(g, config.POINT_PAUSE_S + 0.2, lambda g: (0.5, False), t0=t)
    assert g.state == READY


def test_record_event_only_when_beating_best(unbeatable_opponent):
    g = started_game()
    g.best = 100
    g.update(0, DT, 0.5, False, whistle=True)
    events, _ = play(g, 8, perfect_player)
    assert "hit" in events and "record" not in events


def test_harder_levels_are_faster_with_smaller_paddles():
    e, m, h = (config.LEVELS[k] for k in ("easy", "medium", "hard"))
    assert e["ball_speed"] < m["ball_speed"] < h["ball_speed"]
    assert e["paddle_width"] > m["paddle_width"] > h["paddle_width"]


# ---- points and games -------------------------------------------------------

def never_hits(g):
    return (g.ball.x + 0.5) % 1.0, False


def test_losing_five_points_loses_the_game():
    g = started_game()
    events, _ = play(g, 60, never_hits, auto_serve=True)
    assert events.count("miss") == config.POINTS_TO_WIN
    assert "game_lost" in events and "game_won" not in events
    assert g.state == GAME_OVER and g.winner == "opponent"
    assert (g.player_points, g.opponent_points) == (0, config.POINTS_TO_WIN)


def angling_player(g):
    """Hits every ball off the edge of the paddle, angling it away from center."""
    edge = 0.8 * g.params["paddle_width"] / 2
    in_zone = g.ball.toward_player and g.ball.y >= config.CONTACT_ZONE[0]
    side = 1 if g.ball.x > 0.5 else -1
    return g.ball.x - side * edge, in_zone


def test_angled_shots_beat_the_opponent_and_win_the_game():
    g = started_game("medium")
    events, _ = play(g, 400, angling_player, auto_serve=True)
    assert "game_won" in events
    assert g.state == GAME_OVER and g.winner == "player"
    assert g.player_points == config.POINTS_TO_WIN


def test_winner_point_keeps_the_streak():
    g = started_game("medium")
    events, _ = play(g, 400, angling_player, auto_serve=True)
    first_winner = events.index("winner")
    assert "miss" not in events[:first_winner]
    assert g.streak == events.count("hit")    # never missed, so every hit is in the streak


def test_restart_key_starts_a_fresh_game_but_keeps_the_record():
    g = started_game("hard")
    play(g, 60, never_hits, auto_serve=True)
    g.best = 7
    assert g.state == GAME_OVER
    assert g.update(100, DT, 0.5, False, restart=True) == ["new_game"]
    assert g.state == READY and g.level == "hard"
    assert (g.player_points, g.opponent_points, g.streak, g.best) == (0, 0, 0, 7)


def test_start_tag_after_game_over_starts_new_game():
    g = started_game()
    play(g, 60, never_hits, auto_serve=True)
    assert g.update(100, DT, 0.5, False, tag=config.START_TAG_ID) == ["new_game"]
    assert g.state == READY


def test_level_tag_works_after_game_over():
    g = started_game("easy")
    play(g, 60, never_hits, auto_serve=True)
    g.update(100, DT, 0.5, False, tag=3)
    assert g.level == "hard" and g.state == GAME_OVER


def test_no_serving_after_game_over():
    g = started_game()
    play(g, 60, never_hits, auto_serve=True)
    assert "serve" not in g.update(100, DT, 0.5, False, whistle=True)
    assert g.state == GAME_OVER


# ---- PD opponent ------------------------------------------------------------

def test_opponent_speed_is_capped():
    paddle = PDPaddle(0.0)
    paddle.target = 1.0
    for _ in range(30):
        paddle.update(DT, kp=1000, kd=0, vmax=0.3)
        assert abs(paddle.v) <= 0.3 + 1e-9


@pytest.mark.parametrize("level", list(config.LEVELS))
def test_pd_opponent_settles_on_target(level):
    p = config.LEVELS[level]
    paddle = PDPaddle(0.5)
    paddle.target = 0.85
    for _ in range(int(3 / DT)):
        paddle.update(DT, p["kp"], p["kd"])
    assert abs(paddle.x - 0.85) < 0.01


def test_hard_opponent_reacts_faster_than_easy():
    def time_to_reach(level):
        p = config.LEVELS[level]
        paddle = PDPaddle(0.5)
        paddle.target = 0.85
        for i in range(1000):
            paddle.update(DT, p["kp"], p["kd"])
            if abs(paddle.x - 0.85) < 0.03:
                return i
        return 1000
    assert time_to_reach("hard") < time_to_reach("easy")


# ---- IMU features -----------------------------------------------------------

def test_swing_windows_look_different_from_idle():
    rng = np.random.default_rng(0)
    swing = np.mean([window_features(synthetic_window(1, rng))[4] for _ in range(50)])
    idle = np.mean([window_features(synthetic_window(0, rng))[4] for _ in range(50)])
    assert swing > 3 * idle   # peak rotation speed


def test_window_features_needs_enough_samples():
    assert window_features(np.zeros((2, 6))) is None


# ---- whistle ----------------------------------------------------------------

def tone(freq, seconds=0.05, amp=0.2):
    t = np.arange(int(config.MIC_SAMPLE_RATE * seconds))[:config.MIC_BLOCK] / config.MIC_SAMPLE_RATE
    return amp * np.sin(2 * np.pi * freq * t)


def test_pure_tone_is_whistle_like():
    like, hz, ton, _ = analyze_block(tone(1800))
    assert like and abs(hz - 1800) < 30


def test_noise_is_not_whistle_like():
    noise = np.random.default_rng(0).normal(0, 0.2, config.MIC_BLOCK)
    assert not analyze_block(noise)[0]


def test_whistle_fires_once_after_enough_blocks():
    det = WhistleDetector()
    fired = [det.feed(tone(1500)) for _ in range(config.WHISTLE_MIN_BLOCKS + 5)]
    assert fired.count(True) == 1
    assert fired.index(True) == config.WHISTLE_MIN_BLOCKS - 1


# ---- small helpers ----------------------------------------------------------

def test_score_is_a_float_string():
    assert format_score(7) == "7.0"
    assert float(format_score(0)) == 0.0


def test_camera_x_is_mirrored_and_clipped():
    lo, hi = config.CAMERA_X_RANGE
    assert camera_x_to_table(hi) == pytest.approx(0.0)   # right side of raw image = your left
    assert camera_x_to_table(lo) == pytest.approx(1.0)
    assert camera_x_to_table(0.5) == pytest.approx(0.5)
    assert camera_x_to_table(0.0) == 1.0 and camera_x_to_table(1.0) == 0.0
