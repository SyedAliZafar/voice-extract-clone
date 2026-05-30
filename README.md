# voice-clone-xtts

Extract a speaker's voice from any MP4 video, then synthesize new speech in that voice using Coqui XTTS v2.

## How It Works

| Step | Script | Tool | Description |
|------|--------|------|-------------|
| 1 | `scripts/extract.py` | ffmpeg | Strip audio from MP4 → WAV (24kHz, mono) |
| 2 *(optional)* | `scripts/isolate.py` | demucs | Remove background music/noise, keep vocals |
| 3 | `scripts/synthesize.py` | Coqui XTTS v2 | Clone voice + speak your custom text |

## Project Structure

```
voice-clone-xtts/
├── app.py              # Gradio web UI
├── main.py             # CLI pipeline (all 3 steps)
├── scripts/
│   ├── extract.py      # Step 1: MP4 → WAV via ffmpeg
│   ├── isolate.py      # Step 2: vocal isolation via demucs
│   └── synthesize.py   # Step 3: voice cloning via XTTS v2
├── audio/
│   ├── extracted.wav   # Output of step 1
│   └── vocals.wav      # Output of step 2 (if run)
├── output/
│   └── result.wav      # Final synthesized audio
├── video/              # Place your source MP4 here
└── pyproject.toml
```

## Requirements

- Python 3.9–3.11
- [ffmpeg](https://ffmpeg.org/download.html) installed and on PATH
- ~4 GB VRAM for GPU inference (falls back to CPU automatically)
- ~1.8 GB disk space for XTTS v2 model download (first run only)

## Setup

```bash
# Using uv (recommended)
pip install uv
uv sync

# Or plain pip
pip install -e .
```

## Usage

### CLI

```bash
# Basic — extract voice from video and speak custom text
python main.py --input video/sample.mp4 --text "Your script goes here."

# With vocal isolation (use when video has background music)
python main.py --input video/sample.mp4 --text "Your script." --isolate

# Read script from a text file
python main.py --input video/sample.mp4 --text-file script.txt

# Use a custom voice reference WAV
python main.py --input video/sample.mp4 --text "Hello." --ref audio/my_voice.wav

# Output as MP4
python main.py --input video/sample.mp4 --text "Hello." --output output/result.mp4
```

### Web UI

```bash
uv run python app.py
```

Opens a Gradio interface at `http://localhost:7860`. Type or upload a `.txt` script and click **Generate**.

## Voice Sample Tips

| Sample Length | Quality |
|---|---|
| < 3 seconds | Poor — avoid |
| 6–10 seconds | Good |
| 10–30 seconds | Best |
| > 30 seconds | Diminishing returns |

- Use a clean sample with no background noise or music
- One speaker only
- Neutral tone works best as a base

## Supported Languages

`en` `es` `fr` `de` `it` `pt` `pl` `tr` `ru` `nl` `cs` `ar` `zh-cn` `ja` `ko` `hu`

## Troubleshooting

| Problem | Fix |
|---|---|
| `ffmpeg not found` | Install ffmpeg and ensure it's on your PATH |
| XTTS model not downloading | Check internet connection; model is ~1.8 GB |
| Output sounds robotic | Use a longer, cleaner voice sample (10–20s) |
| CUDA out of memory | Set `CUDA_VISIBLE_DEVICES=""` to force CPU |
| Demucs output not found | Check `audio/htdemucs/<stem>/vocals.wav` manually |
