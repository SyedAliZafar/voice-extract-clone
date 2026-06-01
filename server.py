"""
Flask server for the ExtractVoice UI (index.html).

Endpoints:
  GET  /                    → serves index.html
  POST /upload              → saves file, returns {uid, name, size}
  GET  /run?uid=&mode=      → SSE stream of pipeline progress events
  GET  /download            → sends the processed WAV for download
"""
import json, shutil, uuid
from pathlib import Path
from flask import Flask, request, send_file, jsonify, Response

AUDIO_DIR  = Path("audio")
OUTPUT_DIR = Path("output")
UPLOAD_DIR = Path("uploads")

app = Flask(__name__)


@app.get("/")
def index():
    return send_file(Path("index.html").resolve())


@app.post("/upload")
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="No file provided"), 400
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    uid  = uuid.uuid4().hex[:8]
    dest = UPLOAD_DIR / f"{uid}_{Path(f.filename).name}"
    f.save(dest)
    return jsonify(uid=uid, name=f.filename, size=dest.stat().st_size)


@app.get("/run")
def run():
    uid  = request.args.get("uid", "")
    mode = request.args.get("mode", "vocals")

    matches = list(UPLOAD_DIR.glob(f"{uid}_*")) if uid else []
    src     = matches[0] if matches else None

    def stream():
        from scripts.extract import extract_audio, check_ffmpeg
        from scripts.isolate import isolate_vocals

        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        extracted = AUDIO_DIR / "extracted.wav"
        isolated  = AUDIO_DIR / "vocals.wav"
        out       = OUTPUT_DIR / "result.wav"

        def ev(d): return f"data: {json.dumps(d)}\n\n"

        try:
            # ── Step 1: Extract ───────────────────────────────────────
            yield ev({"step": 1, "status": "processing"})
            if src is None:
                yield ev({"error": "Uploaded file not found — please re-upload."})
                return
            video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
            if src.suffix.lower() in video_exts:
                if not check_ffmpeg():
                    yield ev({"error": "ffmpeg not found — install it and add to PATH."})
                    return
                extract_audio(src, extracted)
            else:
                shutil.copy2(src, extracted)
            ref = extracted
            yield ev({"step": 1, "status": "complete"})

            # ── Step 2: Isolate ───────────────────────────────────────
            yield ev({"step": 2, "status": "processing"})
            if mode in ("vocals", "stems"):
                isolate_vocals(ref, isolated)
                ref = isolated
            yield ev({"step": 2, "status": "complete"})

            # ── Step 3: Encode final output ───────────────────────────
            yield ev({"step": 3, "status": "processing"})
            shutil.copy2(ref, out)
            yield ev({"step": 3, "status": "complete",
                      "size": out.stat().st_size,
                      "download": "/download"})

        except Exception as exc:
            yield ev({"error": str(exc)})

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/download")
def download():
    out = OUTPUT_DIR / "result.wav"
    if not out.exists():
        return jsonify(error="No output yet"), 404
    return send_file(
        out.resolve(),
        as_attachment=True,
        download_name="vocals_isolated.wav",
        mimetype="audio/wav",
    )


if __name__ == "__main__":
    print("ExtractVoice → http://localhost:7860")
    app.run(host="0.0.0.0", port=7860, threaded=True)
