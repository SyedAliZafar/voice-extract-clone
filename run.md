# Run

## Setup (once)

Create and activate a conda environment with Python 3.11 (TTS requires `<3.12`):
```
conda create -n voice-clone python=3.11 -y
conda activate voice-clone
```

Install dependencies:
```
uv sync
uv pip install "torch==2.4.0" "torchaudio==2.4.0" "transformers==4.36.2"
```

## Run the pipeline (CMD)

Put your script in `test.txt`, then run:

```cmd
uv run python main.py --input "video/The_Silent_Departure.mp4" --text-file test.txt --output "output/result.wav"
```

Output will be at `output/result.wav`.

## Run with vocal isolation (removes background music)

```cmd
uv run python main.py --input "video/The_Silent_Departure.mp4" --text-file test.txt --output "output/result.wav" --isolate
```

Use this if the source video has background music — gives XTTS a cleaner voice sample.

## Use a custom voice reference

```cmd
uv run python main.py --input "video/The_Silent_Departure.mp4" --text-file test.txt --output "output/result.wav" --ref audio/vocals.wav
```

## Launch UI

```cmd
uv run python app.py
```

Then open http://localhost:7860 in your browser.
