import json
import time
import paho.mqtt.client as mqtt

BROKER = "localhost"
PORT = 1883

TOPIC = "agrosmart/devices/ESP32-001/command"

payload = {
    "command_id": "TEST-001",
    "device_id": "ESP32-001",
    "action": "SPRAY",
    "bottle": 2,
    "target_volume_ml": 150
}

message = json.dumps(payload)

print("Sending MQTT message:")
print(message)

client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2,
    client_id="AgroSmart-TestPublisher"
)

client.connect(BROKER, PORT, 60)

result = client.publish(
    TOPIC,
    message,
    qos=1
)

result.wait_for_publish()

print("\nMQTT message published successfully.")

client.disconnect()