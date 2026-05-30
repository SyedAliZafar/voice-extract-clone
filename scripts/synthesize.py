from pathlib import Path


def synthesize(voice_reference: Path, text: str, output_path: Path, language: str = "en") -> Path:
    from TTS.api import TTS

    voice_reference = Path(voice_reference)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

    tts.tts_to_file(
        text=text,
        speaker_wav=str(voice_reference),
        language=language,
        file_path=str(output_path),
    )

    print(f"Synthesized audio saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 4:
        print("Usage: python synthesize.py voice_reference.wav "
              '"Text to synthesize" output.wav')
        sys.exit(2)
    voice = Path(sys.argv[1])
    text = sys.argv[2]
    out = Path(sys.argv[3])
    synthesize(voice, text, out)
