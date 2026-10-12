"""
The paddle: the LEGO Double Motor held in your hand.

Two interchangeable versions with the same methods, so the game never
needs to know which one it has:

    LegoPaddle  -- the real hub over Bluetooth. Streams its IMU into a
                   buffer and turns its two motors into a rumble pack.
    SimPaddle   -- no hardware. Press SPACE to swing; it generates IMU
                   readings shaped like a real swing (see imu_swing.py).
                   Haptics become a screen shake plus laptop beeps.

Methods both provide:
    imu_window(now)   -> (N, 6) array of recent IMU readings
    imu_since(t)      -> [(timestamp, reading), ...] newer than t (for recording)
    swing()           -> SimPaddle only: fake a swing (bound to SPACE)
    haptic(event)     -> event is "hit", "miss", "record" or "serve"
    shake_offset(now) -> pixels to jiggle the screen by (SimPaddle only)
    close()
"""

import queue
import random
import threading
import time

import numpy as np

import config
from imu_swing import ImuBuffer, random_gravity, random_swing, synthetic_sample

HAPTIC_EVENTS = ("hit", "miss", "record", "serve")


class LegoPaddle:
    is_sim = False

    def __init__(self):
        import legoeducation as le
        from lelib import doubleMotor

        self._le = le
        color = config.HUB_CARD_COLOR
        card_color = getattr(le, f"LEGO_COLOR_{color.upper()}") if color else None
        card = f"{color} card {config.HUB_CARD_SERIAL}" if color else "any card"
        print(f"Connecting to the Double Motor ({card})...")
        for attempt in range(1, 4):
            self.dm = doubleMotor()
            try:
                self.dm.connect(card_serial=config.HUB_CARD_SERIAL, card_color=card_color)
                break
            except ConnectionError:
                print(f"  attempt {attempt}/3: no Double Motor found for {card}")
                if attempt < 3:
                    time.sleep(2)
        else:
            raise SystemExit(
                "\nCouldn't connect to the Double Motor. Check that:\n"
                f"  - the hub was paired by scanning the {card} (see config.py)\n"
                "  - the hub is on (press its button -- it sleeps after a few idle minutes)\n"
                "  - no other script or LEGO app is still connected to it (close them)\n"
                "  - Bluetooth is on (toggling it off/on in Windows Settings often helps)\n"
                "Or play without it:  python game.py --sim")
        self.dm.device_notification_request(config.IMU_NOTIFY_MS)
        print("Connected.")

        self._imu = ImuBuffer()
        self._running = True
        self._haptics = queue.Queue(maxsize=1)
        threading.Thread(target=self._imu_loop, daemon=True).start()
        threading.Thread(target=self._haptic_loop, daemon=True).start()

    def _imu_loop(self):
        """Copy the hub's latest IMU notification into our buffer ~every 20 ms.
        The library updates dm.imu_device in the background as notifications
        arrive; we just timestamp and save them."""
        last = None
        while self._running:
            d = self.dm.imu_device
            sample = (d.accelerometerX, d.accelerometerY, d.accelerometerZ,
                      d.gyroscopeX, d.gyroscopeY, d.gyroscopeZ)
            if sample != last and not any(v != v for v in sample):   # skip repeats and NaN
                self._imu.add(time.monotonic(), sample)
                last = sample
            time.sleep(config.IMU_NOTIFY_MS / 2000)

    def imu_window(self, now):
        return self._imu.window(now)

    def imu_since(self, t):
        return self._imu.since(t)

    def swing(self):
        pass   # real swings come from your arm

    def shake_offset(self, now):
        return 0, 0

    # -- haptics -----------------------------------------------------------

    def haptic(self, event):
        """Queue a rumble pattern. If one is already playing, drop this one
        rather than letting a backlog of rumbles build up behind the game."""
        try:
            self._haptics.put_nowait(event)
        except queue.Full:
            pass

    def _haptic_loop(self):
        while self._running:
            try:
                event = self._haptics.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                getattr(self, f"_rumble_{event}")()
            except Exception as exc:
                print(f"Haptic '{event}' failed: {exc}")

    def _buzz(self, speed, pulses, on_s, off_s=0.0):
        """Flip both motors back and forth fast. With nothing attached to the
        shafts, the reaction torque is what you feel in your hand."""
        tank = self.dm.movement_move_tank
        for i in range(pulses):
            s = speed if i % 2 == 0 else -speed
            tank(s, -s, blocking=False)
            time.sleep(on_s)
            if off_s:
                tank(0, 0, blocking=False)
                time.sleep(off_s)
        self.dm.movement_stop(blocking=False)

    def _rumble_hit(self):
        # a sharp "thock": two hard, short kicks
        self.dm.light_color(self._le.LEGO_COLOR_GREEN, blocking=False)
        self._buzz(100, pulses=2, on_s=0.04)

    def _rumble_miss(self):
        # a long, droning buzz
        self.dm.light_color(self._le.LEGO_COLOR_RED, blocking=False)
        self.dm.beep(frequency=220, blocking=False)
        self._buzz(60, pulses=12, on_s=0.05)

    def _rumble_record(self):
        # new personal best: triple pulse + a rising beep
        le = self._le
        self.dm.light_color(le.LEGO_COLOR_GREEN, pattern=le.LIGHT_PATTERN_PULSE, blocking=False)
        self.dm.beep(pattern=le.SOUND_PATTERN_BEEP_TRIPLE, frequency=880, blocking=False)
        self._buzz(100, pulses=3, on_s=0.06, off_s=0.08)

    def _rumble_serve(self):
        self.dm.light_color(self._le.LEGO_COLOR_BLUE, blocking=False)
        self._buzz(50, pulses=1, on_s=0.05)

    def close(self):
        self._running = False
        try:
            self.dm.movement_stop()
            self.dm.disconnect()
        except Exception:
            pass


class SimPaddle:
    """Stand-in for the hub. Generates a synthetic IMU stream at the hub's real rate in the
    background; swing() injects a swing into it. The real swing classifier
    still has to recognise it -- the sim does NOT tell the game 'this was a
    swing', it only produces IMU numbers."""

    is_sim = True
    RATE_HZ = config.IMU_RATE_HZ

    def __init__(self, seed=None):
        self._rng = np.random.default_rng(seed)
        self._imu = ImuBuffer()
        self._gravity = random_gravity(self._rng)
        self._swings = []
        self._lock = threading.Lock()
        self._shake_until = 0.0
        self._shake_px = 0
        self._running = True
        self._t0 = time.monotonic()
        threading.Thread(target=self._imu_loop, daemon=True).start()

    def _imu_loop(self):
        while self._running:
            now = time.monotonic()
            t = now - self._t0
            with self._lock:
                self._swings = [s for s in self._swings if t - s["t0"] < 1.0]
                swings = list(self._swings)
            # a little hand wobble so idle data isn't perfectly still
            wobble = (12.0, 0.7, 0.0)
            self._imu.add(now, synthetic_sample(t, self._gravity, swings, self._rng, wobble))
            time.sleep(1 / self.RATE_HZ)

    def imu_window(self, now):
        return self._imu.window(now)

    def imu_since(self, t):
        return self._imu.since(t)

    def swing(self):
        """Peak arrives ~0.1 s after the key press, like a real swing."""
        t = time.monotonic() - self._t0 + 0.1
        with self._lock:
            self._swings.append(random_swing(t, self._rng))

    def haptic(self, event):
        strength = {"hit": (0.12, 6), "miss": (0.5, 10), "record": (0.6, 14), "serve": (0.08, 3)}
        dur, px = strength[event]
        self._shake_until = time.monotonic() + dur
        self._shake_px = px
        tones = {
            "hit": [(1200, 40)],
            "miss": [(220, 350)],
            "record": [(880, 90), (1100, 90), (1320, 160)],
            "serve": [(660, 50)],
        }[event]
        threading.Thread(target=_play_tones, args=(tones,), daemon=True).start()

    def shake_offset(self, now):
        if now >= self._shake_until:
            return 0, 0
        p = self._shake_px
        return random.randint(-p, p), random.randint(-p, p)

    def close(self):
        self._running = False


def _play_tones(tones):
    try:
        import winsound
        for freq, ms in tones:
            winsound.Beep(freq, ms)
    except Exception:
        print("\a", end="", flush=True)   # not Windows: terminal bell


def make_paddle(sim):
    return SimPaddle() if sim else LegoPaddle()
