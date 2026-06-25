import json
import time
import uuid
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT_DIR / "runtime"

LATEST_COMMAND_FILE = RUNTIME_DIR / "latest_command.json"
LISTENER_STATUS_FILE = RUNTIME_DIR / "listener_status.json"
LISTENER_CONTROL_FILE = RUNTIME_DIR / "listener_control.json"


def write_command_event(
    transcription: str,
    wake_result: dict[str, Any],
    parsed_command: dict[str, Any],
) -> dict[str, Any]:
    RUNTIME_DIR.mkdir(exist_ok=True)

    event = {
        "event_id": str(uuid.uuid4()),
        "created_at": time.time(),
        "transcription": transcription,
        "wake_result": wake_result,
        "parsed_command": parsed_command,
    }

    temporary_file = LATEST_COMMAND_FILE.with_suffix(".tmp")

    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(event, file, ensure_ascii=False, indent=2)

    temporary_file.replace(LATEST_COMMAND_FILE)

    return event


def read_latest_command_event() -> dict[str, Any] | None:
    if not LATEST_COMMAND_FILE.exists():
        return None

    try:
        with open(LATEST_COMMAND_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return None
    except PermissionError:
        return None


def clear_latest_command_event():
    if LATEST_COMMAND_FILE.exists():
        LATEST_COMMAND_FILE.unlink()


def write_listener_status(
    status: str,
    message: str,
    is_active: bool,
    transcription: str | None = None,
) -> dict[str, Any]:
    RUNTIME_DIR.mkdir(exist_ok=True)

    event = {
        "updated_at": time.time(),
        "status": status,
        "message": message,
        "is_active": is_active,
        "transcription": transcription,
    }

    temporary_file = LISTENER_STATUS_FILE.with_suffix(".tmp")

    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(event, file, ensure_ascii=False, indent=2)

    temporary_file.replace(LISTENER_STATUS_FILE)

    return event


def read_listener_status() -> dict[str, Any] | None:
    if not LISTENER_STATUS_FILE.exists():
        return None

    try:
        with open(LISTENER_STATUS_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return None
    except PermissionError:
        return None


def write_listener_pause(
    duration_seconds: float,
    reason: str = "pause_requested",
) -> dict[str, Any]:
    """
    Pauses the live listener for a given duration.

    Used when Kokoro is speaking so the microphone does not hear
    the assistant's own voice.
    """

    RUNTIME_DIR.mkdir(exist_ok=True)

    duration_seconds = max(float(duration_seconds), 0.0)

    event = {
        "updated_at": time.time(),
        "pause_until": time.time() + duration_seconds,
        "reason": reason,
        "duration_seconds": duration_seconds,
    }

    temporary_file = LISTENER_CONTROL_FILE.with_suffix(".tmp")

    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(event, file, ensure_ascii=False, indent=2)

    temporary_file.replace(LISTENER_CONTROL_FILE)

    return event


def read_listener_control() -> dict[str, Any] | None:
    """
    Reads listener control instructions.

    Example:
    {
        "pause_until": 1780000000.0,
        "reason": "kokoro_speaking"
    }
    """

    if not LISTENER_CONTROL_FILE.exists():
        return None

    try:
        with open(LISTENER_CONTROL_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return None
    except PermissionError:
        return None


def clear_listener_control():
    if LISTENER_CONTROL_FILE.exists():
        LISTENER_CONTROL_FILE.unlink()