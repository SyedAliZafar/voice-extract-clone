from pathlib import Path
import argparse
from scripts.extract import extract_audio, check_ffmpeg, wav_to_mp4
from scripts.isolate import isolate_vocals
from scripts.synthesize import synthesize


def main():
    p = argparse.ArgumentParser(description="Run voice extraction -> optional isolation -> synthesize")
    p.add_argument("--input", required=True, help="Path to input MP4")
    p.add_argument("--text", help="Text to synthesize")
    p.add_argument("--text-file", help="Path to a .txt file containing text to synthesize")
    p.add_argument("--output", default="output/result.wav", help="Final output WAV path")
    p.add_argument("--isolate", action="store_true", help="Run vocal isolation (demucs)")
    p.add_argument("--ref", help="Optional path to use as voice reference (overrides extracted/isolated)")
    args = p.parse_args()

    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        print("Error: provide --text or --text-file")
        return

    input_path = Path(args.input)
    extracted = Path("audio/extracted.wav")

    if not check_ffmpeg():
        print("ffmpeg not found on PATH. Please install ffmpeg and retry.")
        return

    print(f"Extracting audio from {input_path} -> {extracted}")
    extract_audio(input_path, extracted)

    ref = None
    if args.isolate:
        isolated = Path("audio/vocals.wav")
        print(f"Isolating vocals -> {isolated}")
        isolate_vocals(extracted, isolated)
        ref = isolated
    else:
        ref = extracted

    if args.ref:
        ref = Path(args.ref)

    print(f"Using voice reference: {ref}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    need_mp4 = out.suffix.lower() == ".mp4"
    synth_out = out.with_suffix(".wav") if need_mp4 else out

    try:
        synthesize(ref, text, synth_out)
        if need_mp4:
            wav_to_mp4(synth_out, out)
            print(f"Converted to MP4: {out}")
    except NotImplementedError as e:
        print(str(e))
        print("Synthesis step is a placeholder. I can implement Coqui XTTS integration next.")


if __name__ == "__main__":
    main()
