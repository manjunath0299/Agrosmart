import json
import random
import time
import uuid

import paho.mqtt.client as mqtt


# ============================================================
# CONFIGURATION
# ============================================================

BROKER_HOST = "localhost"
BROKER_PORT = 1883

DEVICE_ID = "ESP32-001"

COMMAND_TOPIC = f"agrosmart/devices/{DEVICE_ID}/command"
STATUS_TOPIC = f"agrosmart/devices/{DEVICE_ID}/status"


# ============================================================
# SIMULATED HARDWARE STATE
# ============================================================

device_state = {
    "device_id": DEVICE_ID,
    "online": True,
    "pump": False,
    "active_valve": None,
    "dispensed_volume_ml": 0.0,
}

# Simulated liquid level for four bottles.
# These are NOT pesticide concentrations or dosages.
bottle_levels = {
    1: 1000.0,
    2: 1000.0,
    3: 1000.0,
    4: 1000.0,
}


# ============================================================
# MQTT STATUS
# ============================================================

def publish_status(
    client,
    status,
    command_id=None,
    bottle=None,
    target_volume_ml=None,
    dispensed_volume_ml=None,
    message=None,
):
    payload = {
        "device_id": DEVICE_ID,
        "status": status,
        "pump": device_state["pump"],
        "active_valve": device_state["active_valve"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if command_id is not None:
        payload["command_id"] = command_id

    if bottle is not None:
        payload["bottle"] = bottle

    if target_volume_ml is not None:
        payload["target_volume_ml"] = target_volume_ml

    if dispensed_volume_ml is not None:
        payload["dispensed_volume_ml"] = round(
            dispensed_volume_ml,
            1,
        )

    if message:
        payload["message"] = message

    client.publish(
        STATUS_TOPIC,
        json.dumps(payload),
        qos=1,
    )

    print("\n[STATUS]")
    print(json.dumps(payload, indent=2))


# ============================================================
# HARDWARE SIMULATION
# ============================================================

def simulate_spray(
    client,
    command_id,
    bottle,
    target_volume_ml,
):
    print("\n" + "=" * 60)
    print("SPRAY COMMAND RECEIVED")
    print("=" * 60)

    print(f"Device       : {DEVICE_ID}")
    print(f"Command ID   : {command_id}")
    print(f"Bottle       : {bottle}")
    print(f"Target volume: {target_volume_ml} mL")

    # --------------------------------------------------------
    # Validate bottle
    # --------------------------------------------------------

    if bottle not in bottle_levels:
        publish_status(
            client,
            status="FAULT",
            command_id=command_id,
            bottle=bottle,
            target_volume_ml=target_volume_ml,
            message="INVALID_BOTTLE",
        )
        return

    # --------------------------------------------------------
    # Validate target volume
    # --------------------------------------------------------

    if target_volume_ml <= 0:
        publish_status(
            client,
            status="FAULT",
            command_id=command_id,
            bottle=bottle,
            target_volume_ml=target_volume_ml,
            message="INVALID_TARGET_VOLUME",
        )
        return

    # --------------------------------------------------------
    # Check bottle liquid
    # --------------------------------------------------------

    if bottle_levels[bottle] < target_volume_ml:
        publish_status(
            client,
            status="LOW_LIQUID",
            command_id=command_id,
            bottle=bottle,
            target_volume_ml=target_volume_ml,
            message="INSUFFICIENT_LIQUID",
        )
        return

    # --------------------------------------------------------
    # Select valve
    # --------------------------------------------------------

    device_state["active_valve"] = bottle

    publish_status(
        client,
        status="VALVE_OPEN",
        command_id=command_id,
        bottle=bottle,
        target_volume_ml=target_volume_ml,
        dispensed_volume_ml=0,
        message=f"Valve {bottle} opened",
    )

    time.sleep(1)

    # --------------------------------------------------------
    # Start pump
    # --------------------------------------------------------

    device_state["pump"] = True
    device_state["dispensed_volume_ml"] = 0.0

    publish_status(
        client,
        status="SPRAYING",
        command_id=command_id,
        bottle=bottle,
        target_volume_ml=target_volume_ml,
        dispensed_volume_ml=0,
        message="Pump started",
    )

    # --------------------------------------------------------
    # Simulated flow sensor
    # --------------------------------------------------------

    dispensed = 0.0

    while dispensed < target_volume_ml:

        # Simulate approximately 20 mL/sec flow.
        flow_increment = random.uniform(15.0, 25.0)

        dispensed += flow_increment

        if dispensed > target_volume_ml:
            dispensed = target_volume_ml

        device_state["dispensed_volume_ml"] = dispensed

        # Consume liquid from selected bottle.
        bottle_levels[bottle] -= flow_increment

        if bottle_levels[bottle] < 0:
            bottle_levels[bottle] = 0

        publish_status(
            client,
            status="SPRAYING",
            command_id=command_id,
            bottle=bottle,
            target_volume_ml=target_volume_ml,
            dispensed_volume_ml=dispensed,
        )

        time.sleep(0.5)

    # --------------------------------------------------------
    # Stop pump
    # --------------------------------------------------------

    device_state["pump"] = False

    publish_status(
        client,
        status="PUMP_STOPPED",
        command_id=command_id,
        bottle=bottle,
        target_volume_ml=target_volume_ml,
        dispensed_volume_ml=dispensed,
        message="Target volume reached",
    )

    time.sleep(0.5)

    # --------------------------------------------------------
    # Close valve
    # --------------------------------------------------------

    device_state["active_valve"] = None

    publish_status(
        client,
        status="VALVE_CLOSED",
        command_id=command_id,
        bottle=bottle,
        target_volume_ml=target_volume_ml,
        dispensed_volume_ml=dispensed,
        message=f"Valve {bottle} closed",
    )

    # --------------------------------------------------------
    # Completed
    # --------------------------------------------------------

    publish_status(
        client,
        status="COMPLETED",
        command_id=command_id,
        bottle=bottle,
        target_volume_ml=target_volume_ml,
        dispensed_volume_ml=dispensed,
        message="Spray operation completed successfully",
    )

    print("\nSPRAY COMPLETED")
    print(f"Bottle remaining: {bottle_levels[bottle]:.1f} mL")


# ============================================================
# MQTT CALLBACK
# ============================================================

def on_connect(client, userdata, flags, reason_code, properties=None):

    print("\n" + "=" * 60)
    print("SIMULATED ESP32 CONNECTED")
    print("=" * 60)

    print(f"Device ID : {DEVICE_ID}")
    print(f"Broker    : {BROKER_HOST}:{BROKER_PORT}")
    print(f"Command   : {COMMAND_TOPIC}")
    print(f"Status    : {STATUS_TOPIC}")

    client.subscribe(
        COMMAND_TOPIC,
        qos=1,
    )

    publish_status(
        client,
        status="ONLINE",
        message="Simulated ESP32 is ready",
    )


def on_disconnect(client, userdata, disconnect_flags, reason_code, properties=None):
    print("\n[MQTT] Disconnected")


def on_message(client, userdata, msg):

    print("\n" + "-" * 60)
    print("MQTT COMMAND RECEIVED")
    print("-" * 60)

    try:

        payload = json.loads(
            msg.payload.decode("utf-8")
        )

        print(json.dumps(payload, indent=2))

    except json.JSONDecodeError:

        print("[ERROR] Invalid JSON command")

        publish_status(
            client,
            status="FAULT",
            message="INVALID_JSON_COMMAND",
        )

        return

    # --------------------------------------------------------
    # Validate command
    # --------------------------------------------------------

    action = payload.get("action")

    if action != "SPRAY":

        publish_status(
            client,
            status="FAULT",
            message=f"UNSUPPORTED_ACTION: {action}",
        )

        return

    # --------------------------------------------------------
    # Extract command
    # --------------------------------------------------------

    command_id = payload.get(
        "command_id",
        str(uuid.uuid4()),
    )

    device_id = payload.get("device_id")

    bottle = payload.get("bottle")
    target_volume_ml = payload.get("target_volume_ml")

    # --------------------------------------------------------
    # Device ID check
    # --------------------------------------------------------

    if device_id != DEVICE_ID:

        print(
            f"[WARNING] Command intended for {device_id}"
        )

        return

    # --------------------------------------------------------
    # Type validation
    # --------------------------------------------------------

    try:
        bottle = int(bottle)
        target_volume_ml = float(target_volume_ml)

    except (TypeError, ValueError):

        publish_status(
            client,
            status="FAULT",
            command_id=command_id,
            message="INVALID_COMMAND_PARAMETERS",
        )

        return

    # --------------------------------------------------------
    # Execute spray
    # --------------------------------------------------------

    simulate_spray(
        client,
        command_id,
        bottle,
        target_volume_ml,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nStarting AgroSmart simulated ESP32...")

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=DEVICE_ID,
    )

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    print(
        f"Connecting to MQTT broker "
        f"{BROKER_HOST}:{BROKER_PORT}..."
    )

    client.connect(
        BROKER_HOST,
        BROKER_PORT,
        keepalive=60,
    )

    try:

        client.loop_forever()

    except KeyboardInterrupt:

        print("\nStopping simulated ESP32...")

        device_state["pump"] = False
        device_state["active_valve"] = None

        client.disconnect()


if __name__ == "__main__":
    main()