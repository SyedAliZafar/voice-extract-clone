# CLAUDE.md — Voice Extraction & Cloning Project

## Project Overview
Extract voice audio from an MP4 file, then use that voice sample to synthesize new speech from a custom text script using AI voice cloning.

## Stack
- **Python 3.9+**
- **ffmpeg** — audio extraction from MP4
- **Coqui XTTS v2** — voice cloning and TTS synthesis
- **demucs** (optional) — vocal isolation if background noise is present

## Project Structure
```
voice-clone/
├── CLAUDE.md          # This file — instructions for Claude
├── CONTEXT.md         # Project context and pipeline details
├── input/
│   └── video.mp4      # Source MP4 file
├── audio/
│   ├── extracted.wav  # Raw audio extracted from MP4
│   └── vocals.wav     # (Optional) isolated vocals after demucs
├── output/
│   └── result.wav     # Final synthesized speech
├── scripts/
│   ├── extract.py     # Step 1: Extract audio from MP4
│   ├── isolate.py     # Step 2 (optional): Isolate vocals
│   └── synthesize.py  # Step 3: Clone voice + generate speech
├── requirements.txt
└── main.py            # Run full pipeline end-to-end
```

## Pipeline (3 Steps)
1. **Extract** — Pull audio from MP4 using ffmpeg
2. **Isolate** *(optional)* — Strip background music/noise with demucs
3. **Synthesize** — Feed voice sample + custom text to XTTS v2

## Setup Instructions
```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python main.py --input input/video.mp4 --text "Your custom script goes here."
```

## Commands for Claude
When working with this project, Claude should:
- Always check that `ffmpeg` is installed before running extract steps
- Use `audio/vocals.wav` as the speaker reference if isolation was run, otherwise use `audio/extracted.wav`
- Keep voice samples between **6–30 seconds** for best XTTS cloning quality
- Output audio as **WAV, 24kHz, mono** unless specified otherwise
- Never hardcode file paths — always use `pathlib.Path`

## Notes
- XTTS v2 requires ~4GB of VRAM for GPU inference; falls back to CPU automatically
- First run will download the XTTS v2 model (~1.8 GB)
- Supported languages: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, ko, hu
