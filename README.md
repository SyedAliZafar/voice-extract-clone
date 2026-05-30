# Voice Extraction & Cloning (starter)

Quick starter pipeline to extract audio from an MP4, optionally isolate vocals with `demucs`, and synthesize speech using Coqui XTTS.

Usage (example):

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py --input input/video.mp4 --text "Hello world" --output output/result.wav
```

Notes:
- `scripts/synthesize.py` is a placeholder; I can implement Coqui XTTS integration next.
- `extract.py` forces output to WAV, 24kHz, mono.
- If `demucs` is not installed, `isolate.py` will copy the extracted file as a fallback.
