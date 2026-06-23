# Local STT Voice-Controlled Dashboard

This project is a local prototype for controlling a Streamlit dashboard using voice commands.

The goal is to help a user who cannot use their arms navigate through a dashboard using speech.

The system works locally with:

- Streamlit for the dashboard
- faster-whisper for local Speech-To-Text
- a custom command parser
- a continuous voice listener
- a wake phrase: **"Ok Jack"**

---

## Project Flow

```txt
Microphone
↓
Continuous listener
↓
Detects "Ok Jack"
↓
Transcribes the command locally with Whisper
↓
Parses the text into a dashboard action
↓
Streamlit dashboard updates automatically
```

Example:

```txt
User says:
Ok Jack, affiche les ventes par région

Whisper may transcribe:
Ok Jacques, affiche l'évente par région

System understands:
show_chart / ventes / region

Dashboard action:
Displays sales by region
```

---

## Project Structure

```txt
stt_dashboard_voice/
│
├── dashboard/
│   └── app.py
│
├── src/
│   ├── audio_utils.py
│   ├── command_bus.py
│   ├── command_parser.py
│   ├── continuous_listener.py
│   ├── dashboard_controller.py
│   ├── main.py
│   ├── stt_engine.py
│   └── wake_word.py
│
├── audio_samples/
│   └── test.wav
│
├── runtime/
│   └── generated at runtime, not committed
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Requirements

Recommended:

```txt
Python 3.10 or higher
Windows PowerShell
Microphone access enabled
```

The project was developed and tested on Windows.

---

## Installation

Clone the repository:

```powershell
git clone https://github.com/LucasRambert4/Projet-IA-Generative.git
cd Projet-IA-Generative
```

Checkout the working branch:

```powershell
git checkout v1.1-wake-word-ok-jack
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## Running the Project

You need to open **two terminals**.

---

### Terminal 1 — Start the Streamlit dashboard

From the project root:

```powershell
.venv\Scripts\Activate.ps1
streamlit run dashboard/app.py
```

This opens the dashboard in the browser.

---

### Terminal 2 — Start the continuous voice listener

Open another PowerShell terminal.

From the project root:

```powershell
.venv\Scripts\Activate.ps1
python src/continuous_listener.py
```

Expected terminal output:

```txt
Loading local STT model...

Continuous voice listener started.
Say: Ok Jack, affiche les ventes par région
Press CTRL + C to stop.
```

The first launch may take longer because the Whisper model may need to be downloaded.

---

## How to Test

Once both terminals are running, say one of these commands clearly:

```txt
Ok Jack, affiche les ventes par région
```

```txt
Ok Jack, va à la page ventes
```

```txt
Ok Jack, va à la page clients
```

```txt
Ok Jack, réinitialise les filtres
```

You can also test with imperfect pronunciation or transcription:

```txt
Ok Jacques, affiche l'évente par région
```

The system should still understand the command.

---

## Expected Result

In the listener terminal, you should see something like:

```txt
Transcription: Ok Jacques, réinitialise les filtres.
Wake word detected.
Command: reinitialise les filtres
Parsed: {'intent': 'reset_filters', 'raw_text': 'reinitialise les filtres'}
```

In the Streamlit dashboard, the page or chart should update automatically.

---

## Wake Word Behavior

The dashboard only reacts when the command starts with:

```txt
Ok Jack
```

Accepted variations include:

```txt
Ok Jack
Okay Jack
Ok Jacques
Okay Jacques
Ok Jak
```

This avoids executing accidental speech.

---

## Two Command Modes

The system supports two patterns.

### 1. Wake word and command together

```txt
Ok Jack, affiche les ventes par région
```

### 2. Wake word first, command after

Say:

```txt
Ok Jack
```

Then quickly say:

```txt
Va à la page ventes
```

The listener waits a few seconds for the follow-up command.

---

## Current Supported Commands

| Voice command | Expected action |
|---|---|
| Ok Jack, va à la page ventes | Opens the sales page |
| Ok Jack, va à la page clients | Opens the clients page |
| Ok Jack, affiche les ventes par région | Shows sales by region |
| Ok Jack, réinitialise les filtres | Resets filters and returns to summary |

---

## Technical Notes

The system is separated into modules:

| File | Role |
|---|---|
| `stt_engine.py` | Local Speech-To-Text using faster-whisper |
| `wake_word.py` | Detects the "Ok Jack" trigger |
| `command_parser.py` | Converts text into dashboard commands |
| `continuous_listener.py` | Continuously listens to the microphone |
| `command_bus.py` | Sends commands from the listener to Streamlit |
| `dashboard_controller.py` | Applies commands to the dashboard state |
| `dashboard/app.py` | Streamlit dashboard interface |

---

## Why There Are Two Processes

Streamlit is not ideal for always-on microphone listening.

So the project uses two separate processes:

```txt
1. Streamlit dashboard
2. Continuous local voice listener
```

They communicate through a local JSON file:

```txt
runtime/latest_command.json
```

This file is generated automatically and should not be committed.

---

## Troubleshooting

### The microphone does not work

Check that Windows allows microphone access for the terminal or Python.

Also check your default input device.

---

### The dashboard does not update

Make sure both terminals are running:

```txt
Terminal 1: streamlit run dashboard/app.py
Terminal 2: python src/continuous_listener.py
```

Also check that `runtime/latest_command.json` is being created after a voice command.

---

### Whisper is slow

The current model is:

```txt
base
```

It is more accurate than very small models but may be slower on some machines.

If needed, change this in:

```txt
src/continuous_listener.py
```

From:

```python
MODEL_SIZE = "base"
```

To:

```python
MODEL_SIZE = "small"
```

or for faster but less accurate testing:

```python
MODEL_SIZE = "tiny"
```

---

### The wake word is not detected

Try saying:

```txt
Ok Jack
```

or:

```txt
Ok Jacques
```

clearly at the beginning of the sentence.

Example:

```txt
Ok Jack, affiche les ventes par région
```

Avoid starting directly with the command:

```txt
Affiche les ventes par région
```

This will be ignored because the wake word is missing.

---

## Git Notes

Do not commit:

```txt
.venv/
runtime/
audio_samples/*.wav
audio_samples/*.mp3
audio_samples/*.m4a
```

These should be ignored by `.gitignore`.

---

## Development Status

Current version:

```txt
V1.1 — Continuous voice activation with "Ok Jack"
```

Implemented:

- Local STT with faster-whisper
- Streamlit dashboard
- Voice command parser
- Wake word detection
- Continuous microphone listener
- Local communication between listener and dashboard

Next possible improvements:

- Add command history in the dashboard
- Add a list of available commands in the UI
- Add evaluation metrics
- Integrate into the real dashboard
- Add a local LLM for more flexible command understanding