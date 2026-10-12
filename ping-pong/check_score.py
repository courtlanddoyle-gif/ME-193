"""
Subscribe to the live score topics and print every update -- run this in
a second terminal while playing to prove the score is really being posted.
Shows both the record (ME193/Rogers/Courtland) and the game to 5
(ME193/Rogers/Courtland/game).

    python check_score.py
"""

import time

import config
from mqttlib import MQTTClient


def on_message(topic, payload):
    print(f"[{time.strftime('%H:%M:%S')}] {topic}: {payload}")


client = MQTTClient(broker=config.MQTT_BROKER, port=config.MQTT_PORT)
client.connect()
client.subscribe(config.SCORE_TOPIC, on_message)
client.subscribe(config.GAME_TOPIC, on_message)
print(f"Listening on {config.MQTT_BROKER} / {config.SCORE_TOPIC} and {config.GAME_TOPIC}"
      "  (Ctrl+C to stop)")
try:
    while True:
        time.sleep(0.2)
except KeyboardInterrupt:
    client.disconnect()
