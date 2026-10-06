import json
import uuid
from datetime import datetime

import paho.mqtt.client as mqtt


BROKER_HOST = "localhost"
BROKER_PORT = 1883

DEFAULT_DEVICE_ID = "ESP32-001"


class MQTTService:
    def __init__(
        self,
        broker_host: str = BROKER_HOST,
        broker_port: int = BROKER_PORT,
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port

        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"AgroSmart-FastAPI-{uuid.uuid4().hex[:8]}",
        )

        self.connected = False

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

        self._connect()

    # ---------------------------------------------------------
    # MQTT CONNECTION
    # ---------------------------------------------------------

    def _connect(self):
        try:
            self.client.connect(
                self.broker_host,
                self.broker_port,
                keepalive=60,
            )

            # Run MQTT network processing in background.
            self.client.loop_start()

        except Exception as exc:
            self.connected = False
            print(f"[MQTT] Connection failed: {exc}")

    def _on_connect(
        self,
        client,
        userdata,
        flags,
        reason_code,
        properties=None,
    ):
        if reason_code == 0:
            self.connected = True
            print(
                f"[MQTT] Connected to "
                f"{self.broker_host}:{self.broker_port}"
            )
        else:
            self.connected = False
            print(
                f"[MQTT] Connection rejected: "
                f"{reason_code}"
            )

    def _on_disconnect(
        self,
        client,
        userdata,
        disconnect_flags,
        reason_code,
        properties=None,
    ):
        self.connected = False
        print("[MQTT] Disconnected")

    # ---------------------------------------------------------
    # DEVICE TOPICS
    # ---------------------------------------------------------

    @staticmethod
    def command_topic(device_id: str) -> str:
        return (
            f"agrosmart/devices/"
            f"{device_id}/command"
        )

    @staticmethod
    def status_topic(device_id: str) -> str:
        return (
            f"agrosmart/devices/"
            f"{device_id}/status"
        )

    # ---------------------------------------------------------
    # PUBLISH SPRAY COMMAND
    # ---------------------------------------------------------

    def publish_spray_command(
        self,
        device_id: str,
        bottle: int,
        target_volume_ml: float,
    ) -> dict:

        if not self.connected:
            raise RuntimeError(
                "MQTT broker is not connected."
            )

        command_id = (
            f"SP-"
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
            f"{uuid.uuid4().hex[:6].upper()}"
        )

        payload = {
            "command_id": command_id,
            "device_id": device_id,
            "action": "SPRAY",
            "bottle": bottle,
            "target_volume_ml": target_volume_ml,
        }

        topic = self.command_topic(device_id)

        result = self.client.publish(
            topic,
            json.dumps(payload),
            qos=1,
        )

        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(
                f"MQTT publish failed: {result.rc}"
            )

        print("\n[MQTT] SPRAY COMMAND PUBLISHED")
        print(f"Topic   : {topic}")
        print(
            json.dumps(
                payload,
                indent=2,
            )
        )

        return {
            "published": True,
            "command_id": command_id,
            "device_id": device_id,
            "topic": topic,
            "payload": payload,
        }

    # ---------------------------------------------------------
    # HEALTH
    # ---------------------------------------------------------

    def health(self) -> dict:
        return {
            "connected": self.connected,
            "broker": self.broker_host,
            "port": self.broker_port,
        }


# Singleton used by FastAPI.
mqtt_service = MQTTService()