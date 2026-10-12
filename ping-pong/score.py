"""
Live score over MQTT.

Publishes the record (longest streak of continuous hits) as a floating
point number, e.g. "7.0", to ME193/Rogers/Courtland every time it goes up.
Messages are retained, so anyone who subscribes later immediately gets the
current record instead of waiting for the next one. The record is also
saved to data/best_score.json so it survives restarting the game.

Check it from another terminal (or another laptop):
    python check_score.py
"""

import json
import threading

import config


def load_best():
    try:
        return int(json.loads(config.BEST_SCORE_FILE.read_text())["best"])
    except (FileNotFoundError, KeyError, ValueError):
        return 0


def save_best(best):
    """Never let saving the record interrupt the game -- Box sync can
    briefly lock files on Windows."""
    try:
        config.DATA_DIR.mkdir(exist_ok=True)
        config.BEST_SCORE_FILE.write_text(json.dumps({"best": best}))
    except OSError as exc:
        print(f"Couldn't save record ({exc}); it will be saved next time.")


def format_score(best):
    return f"{float(best):.1f}"


class ScorePublisher:
    """Publishes in a background thread so a slow or unreachable public
    broker never freezes the game."""

    def __init__(self, enabled=True, broker=config.MQTT_BROKER, topic=config.SCORE_TOPIC):
        self.topic = topic
        self.status = "off" if not enabled else "connecting"
        self._client = None
        self._pending = None
        self._wake = threading.Event()
        if enabled:
            threading.Thread(target=self._run, args=(broker,), daemon=True).start()

    def _run(self, broker):
        from mqttlib import MQTTClient
        try:
            client = MQTTClient(broker=broker, port=config.MQTT_PORT)
            client.connect(timeout=5)
            self._client = client
            self.status = f"connected ({broker})"
        except Exception as exc:
            self.status = f"offline: {exc}"
            print(f"MQTT: could not reach {broker}: {exc}  (game still works, score not posted)")
            return
        while True:
            self._wake.wait()
            self._wake.clear()
            message, self._pending = self._pending, None
            if message is not None:
                self._client.publish(self.topic, message, qos=1, retain=True)
                print(f"MQTT: {self.topic} <- {message}")

    def publish(self, best):
        self._pending = format_score(best)
        self._wake.set()

    def close(self):
        if self._client is not None:
            self._client.disconnect()
