import json
import time
import uuid
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT_DIR / "runtime"
LATEST_COMMAND_FILE = RUNTIME_DIR / "latest_command.json"


def write_command_event(
    transcription: str,
    wake_result: dict[str, Any],
    parsed_command: dict[str, Any],
) -> dict[str, Any]:
    """
    Writes the latest voice command event to a JSON file.

    This file is used as a bridge between:
    - the voice listener running in a terminal
    - the Streamlit dashboard running in another terminal
    """

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
    """
    Reads the latest command event from the runtime JSON file.

    Returns None if no command has been written yet.
    """

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
    """
    Deletes the latest command event file.

    This is optional and mostly useful for debugging.
    """

    if LATEST_COMMAND_FILE.exists():
        LATEST_COMMAND_FILE.unlink()