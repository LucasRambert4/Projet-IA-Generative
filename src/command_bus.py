import json
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT_DIR / "runtime"
LATEST_COMMAND_FILE = RUNTIME_DIR / "latest_command.json"


def write_command_event(
    transcription: str,
    wake_result: dict,
    parsed_command: dict
):
    """
    Writes the latest voice command to a local JSON file.

    This acts as a simple communication bridge between:
    - the continuous voice listener
    - the Streamlit dashboard
    """

    RUNTIME_DIR.mkdir(exist_ok=True)

    event = {
        "id": str(time.time_ns()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "transcription": transcription,
        "wake_result": wake_result,
        "parsed_command": parsed_command
    }

    temp_file = LATEST_COMMAND_FILE.with_suffix(".tmp")

    temp_file.write_text(
        json.dumps(event, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    temp_file.replace(LATEST_COMMAND_FILE)


def read_latest_command_event() -> dict | None:
    """
    Reads the latest voice command event.

    Returns None if no command has been detected yet.
    """

    if not LATEST_COMMAND_FILE.exists():
        return None

    try:
        return json.loads(
            LATEST_COMMAND_FILE.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError:
        return None