"""
Subscribe to the live score topic and print every update -- run this in a
second terminal while playing to prove the score is really being posted.

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
print(f"Listening on {config.MQTT_BROKER} / {config.SCORE_TOPIC}  (Ctrl+C to stop)")
try:
    while True:
        time.sleep(0.2)
except KeyboardInterrupt:
    client.disconnect()
