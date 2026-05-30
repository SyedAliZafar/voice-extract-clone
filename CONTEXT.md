# CONTEXT.md — Technical Context & Pipeline

## What This Project Does
Takes any MP4 video, extracts the speaker's voice, and uses it to generate
new speech from your own custom text — making it sound like the same person.

---

## Pipeline Detail

### Step 1 — Extract Audio (`scripts/extract.py`)
**Tool:** `ffmpeg-python`  
**Input:** `input/video.mp4`  
**Output:** `audio/extracted.wav`

```python
import ffmpeg
from pathlib import Path

def extract_audio(input_path: str, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    (
        ffmpeg
        .input(input_path)
        .output(output_path, acodec='pcm_s16le', ar=22050, ac=1)
        .overwrite_output()
        .run(quiet=True)
    )
    print(f"[✓] Audio extracted → {output_path}")
```

**Why WAV?** XTTS v2 and demucs both expect uncompressed PCM audio.  
**Why 22050 Hz mono?** Minimum quality needed for voice cloning; keeps file sizes small.

---

### Step 2 — Isolate Vocals (`scripts/isolate.py`) *(Optional)*
**Tool:** `demucs` (Facebook Research)  
**Input:** `audio/extracted.wav`  
**Output:** `audio/vocals.wav`

Use this step when:
- There is background music in the video
- Multiple speakers are present and you want one
- There is heavy noise or reverb

```bash
# CLI usage
python -m demucs --two-stems=vocals audio/extracted.wav -o audio/
```

```python
# Programmatic usage
import subprocess

def isolate_vocals(input_path: str):
    subprocess.run([
        "python", "-m", "demucs",
        "--two-stems=vocals",
        input_path,
        "-o", "audio/"
    ], check=True)
    print("[✓] Vocals isolated → audio/htdemucs/extracted/vocals.wav")
```

**Note:** demucs outputs to `audio/htdemucs/<filename>/vocals.wav` — copy to `audio/vocals.wav` after.

---

### Step 3 — Voice Cloning + Synthesis (`scripts/synthesize.py`)
**Tool:** Coqui `XTTS v2`  
**Input:** speaker WAV sample + your text script  
**Output:** `output/result.wav`

```python
from TTS.api import TTS
from pathlib import Path

def synthesize(
    text: str,
    speaker_wav: str,
    output_path: str,
    language: str = "en"
):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language=language,
        file_path=output_path
    )
    print(f"[✓] Speech synthesized → {output_path}")
```

**Voice sample tips for best quality:**
| Sample Length | Quality |
|---|---|
| < 3 seconds | Poor — avoid |
| 6–10 seconds | Good |
| 10–30 seconds | Best |
| > 30 seconds | Diminishing returns |

- Sample should be **clean speech** (no music, no noise)
- **One speaker only** in the sample
- **Neutral tone** works best as a base

---

### Full Pipeline (`main.py`)

```python
import argparse
from scripts.extract import extract_audio
from scripts.isolate import isolate_vocals
from scripts.synthesize import synthesize
from pathlib import Path
import shutil

def main():
    parser = argparse.ArgumentParser(description="Voice Extraction & Cloning Pipeline")
    parser.add_argument("--input",    required=True,  help="Path to input MP4 file")
    parser.add_argument("--text",     required=True,  help="Script text to synthesize")
    parser.add_argument("--isolate",  action="store_true", help="Run vocal isolation step")
    parser.add_argument("--lang",     default="en",   help="Language code (default: en)")
    parser.add_argument("--output",   default="output/result.wav")
    args = parser.parse_args()

    # Step 1: Extract
    extract_audio(args.input, "audio/extracted.wav")

    # Step 2: Isolate (optional)
    if args.isolate:
        isolate_vocals("audio/extracted.wav")
        shutil.copy("audio/htdemucs/extracted/vocals.wav", "audio/vocals.wav")
        speaker_ref = "audio/vocals.wav"
    else:
        speaker_ref = "audio/extracted.wav"

    # Step 3: Synthesize
    synthesize(args.text, speaker_ref, args.output, args.lang)
    print(f"\n✅ Done! Output saved to: {args.output}")

if __name__ == "__main__":
    main()
```

---

## Requirements (`requirements.txt`)

```
ffmpeg-python==0.2.0
TTS==0.22.0
demucs==4.0.1
torch>=2.0.0
torchaudio>=2.0.0
```

---

## Hardware Requirements

| Mode | Requirement |
|---|---|
| CPU only | Works, slow (~2–5 min per sentence) |
| GPU (CUDA) | Recommended (RTX 3060+ or better) |
| VRAM needed | ~4 GB for XTTS v2 |
| RAM needed | ~8 GB minimum |

---

## Supported Languages (XTTS v2)
`en` `es` `fr` `de` `it` `pt` `pl` `tr` `ru` `nl` `cs` `ar` `zh-cn` `ja` `ko` `hu`

---

## Common Issues

| Problem | Fix |
|---|---|
| `ffmpeg not found` | Install system ffmpeg: `sudo apt install ffmpeg` or `brew install ffmpeg` |
| XTTS model not downloading | Check internet connection; model is ~1.8 GB |
| Output sounds robotic | Use a longer, cleaner voice sample (10–20s) |
| CUDA out of memory | Set `CUDA_VISIBLE_DEVICES=""` to force CPU |
| Demucs wrong output path | Check `audio/htdemucs/<input_stem>/vocals.wav` |
