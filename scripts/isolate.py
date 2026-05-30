from pathlib import Path
import shutil
import subprocess


def demucs_available() -> bool:
    return shutil.which("demucs") is not None


def isolate_vocals(input_wav: Path, output_wav: Path) -> Path:
    input_wav = Path(input_wav)
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    if demucs_available():
        # Use demucs to separate vocals; this produces a folder with stems.
        cmd = ["demucs", "-n", "htdemucs", "-o", str(output_wav.parent), str(input_wav)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError("demucs failed to isolate vocals")
        # demucs creates a folder structure; attempt to locate vocals.wav
        # This is a best-effort: user may need to adjust.
        candidate = output_wav.parent / "htdemucs" / input_wav.stem / "vocals.wav"
        if candidate.exists():
            candidate.replace(output_wav)
            return output_wav
        else:
            raise FileNotFoundError("demucs ran but vocals file not found")
    else:
        # demucs not available; fallback: copy input to output
        shutil.copy2(str(input_wav), str(output_wav))
        return output_wav


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 3:
        print("Usage: python isolate.py input.wav output_vocals.wav")
        sys.exit(2)
    inpath = Path(sys.argv[1])
    outpath = Path(sys.argv[2])
    out = isolate_vocals(inpath, outpath)
    print(f"Wrote: {out}")
