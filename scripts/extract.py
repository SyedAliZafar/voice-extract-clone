from pathlib import Path
import shutil
import subprocess
import sys


def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def extract_audio(input_path: Path, output_path: Path) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not check_ffmpeg():
        raise EnvironmentError("ffmpeg not found on PATH")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to WAV, 24kHz, mono (pcm_s16le)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        "24000",
        "-ac",
        "1",
        "-vn",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        msg = proc.stderr.decode(errors="ignore")
        raise RuntimeError(f"ffmpeg failed: {msg}")

    return output_path


def wav_to_mp4(wav_path: Path, mp4_path: Path) -> Path:
    wav_path = Path(wav_path)
    mp4_path = Path(mp4_path)
    mp4_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(wav_path),
        "-c:a", "aac",
        "-b:a", "192k",
        str(mp4_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        msg = proc.stderr.decode(errors="ignore")
        raise RuntimeError(f"ffmpeg failed: {msg}")

    wav_path.unlink()
    return mp4_path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract.py input.mp4 output.wav")
        sys.exit(2)
    inpath = Path(sys.argv[1])
    outpath = Path(sys.argv[2])
    out = extract_audio(inpath, outpath)
    print(f"Wrote: {out}")
