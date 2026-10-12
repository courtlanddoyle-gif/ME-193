"""
Live score over MQTT, on two topics:

    ME193/Rogers/Courtland        the record (longest streak of continuous
                                  hits) as a float, e.g. "7.0", every time
                                  it goes up -- the assignment's format
    ME193/Rogers/Courtland/game   the current game to 5, your points first:
                                  "0-0", "2-1", ... "5-3 YOU WIN"

The game score lives on its own subtopic so the record topic only ever
carries a plain number. Messages are retained, so anyone who subscribes
later immediately gets the latest values. The record is also saved to
data/best_score.json so it survives restarting the game.

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


def format_game(player_points, cpu_points, winner=None):
    text = f"{player_points}-{cpu_points}"
    if winner == "player":
        text += " YOU WIN"
    elif winner == "opponent":
        text += " CPU WINS"
    return text


class ScorePublisher:
    """Publishes in a background thread so a slow or unreachable public
    broker never freezes the game."""

    def __init__(self, enabled=True, broker=config.MQTT_BROKER):
        self.status = "off" if not enabled else "connecting"
        self._client = None
        self._pending = {}            # topic -> latest message not yet sent
        self._lock = threading.Lock()
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
            with self._lock:
                pending, self._pending = self._pending, {}
            for topic, message in pending.items():
                self._client.publish(topic, message, qos=1, retain=True)
                print(f"MQTT: {topic} <- {message}")

    def _queue(self, topic, message):
        with self._lock:
            self._pending[topic] = message     # only the newest value per topic matters
        self._wake.set()

    def publish(self, best):
        """The record, as a float."""
        self._queue(config.SCORE_TOPIC, format_score(best))

    def publish_game(self, player_points, cpu_points, winner=None):
        """The current game score."""
        self._queue(config.GAME_TOPIC, format_game(player_points, cpu_points, winner))

    def close(self):
        if self._client is not None:
            self._client.disconnect()
