import shutil
import time
import numpy as np
import scipy.io.wavfile as wavfile
import gradio as gr
from pathlib import Path

from scripts.extract import extract_audio, check_ffmpeg
from scripts.isolate import isolate_vocals, demucs_available
from scripts.synthesize import synthesize

# ── Constants ──────────────────────────────────────────────────────────────────
AUDIO_DIR = Path("audio")
OUTPUT = Path("output/result.wav")
MIC_WAV = AUDIO_DIR / "mic_input.wav"
EXTRACTED = AUDIO_DIR / "extracted.wav"
ISOLATED = AUDIO_DIR / "vocals.wav"

LANGUAGES = [
    ("English", "en"), ("Spanish", "es"), ("French", "fr"),
    ("German", "de"), ("Italian", "it"), ("Portuguese", "pt"),
    ("Polish", "pl"), ("Turkish", "tr"), ("Russian", "ru"),
    ("Dutch", "nl"), ("Czech", "cs"), ("Arabic", "ar"),
    ("Chinese (Mandarin)", "zh-cn"), ("Japanese", "ja"),
    ("Korean", "ko"), ("Hungarian", "hu"),
]

HAS_DEMUCS = demucs_available()
HAS_FFMPEG = check_ffmpeg()


# ── Backend functions (unchanged) ─────────────────────────────────────────────
def _ensure_dirs():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def run_mic(audio_input, text: str, language: str):
    if audio_input is None:
        return None, "Please record your voice first."
    text = text.strip() if text else ""
    if not text:
        return None, "Please enter the text you want to speak."
    _ensure_dirs()
    sample_rate, data = audio_input
    if data.ndim > 1:
        data = data[:, 0]
    wavfile.write(str(MIC_WAV), sample_rate, data.astype(np.int16))
    try:
        synthesize(MIC_WAV, text, OUTPUT, language)
        return str(OUTPUT), "Done — audio generated successfully."
    except Exception as exc:
        return None, f"Error: {exc}"


def run_file(file_upload, isolate: bool, text: str, language: str):
    if file_upload is None:
        return None, "Please upload an MP4 or WAV file."
    text = text.strip() if text else ""
    if not text:
        return None, "Please enter the text you want to speak."
    _ensure_dirs()
    upload_path = Path(file_upload)
    suffix = upload_path.suffix.lower()
    try:
        if suffix == ".mp4":
            if not HAS_FFMPEG:
                return None, "Error: ffmpeg is not installed. Please install it and add it to PATH."
            extract_audio(upload_path, EXTRACTED)
            ref = EXTRACTED
        elif suffix == ".wav":
            shutil.copy2(upload_path, EXTRACTED)
            ref = EXTRACTED
        else:
            return None, f"Unsupported file type: {suffix}. Please upload .mp4 or .wav."
        if isolate and HAS_DEMUCS:
            isolate_vocals(ref, ISOLATED)
            ref = ISOLATED
        synthesize(ref, text, OUTPUT, language)
        return str(OUTPUT), "Done — audio generated successfully."
    except Exception as exc:
        return None, f"Error: {exc}"


# ── UI helpers ─────────────────────────────────────────────────────────────────
def _build_stats_html(audio_path, elapsed: float) -> str:
    if not audio_path:
        return ""
    duration_str = "—"
    try:
        sr, data = wavfile.read(str(audio_path))
        duration_str = f"{data.shape[0] / sr:.1f}s"
    except Exception:
        pass
    return (
        f'<div class="vc-gen-info">'
        f'<span>Generated in {elapsed:.1f}s &middot; {duration_str} audio</span>'
        f'</div>'
    )


def run_unified(mic_input, file_input, isolate: bool, text: str, language: str):
    t0 = time.perf_counter()
    if mic_input is not None:
        audio_path, status = run_mic(mic_input, text, language)
    elif file_input is not None:
        audio_path, status = run_file(file_input, isolate, text, language)
    else:
        return None, "Please record your voice or upload an audio file.", ""
    elapsed = time.perf_counter() - t0
    return audio_path, status, _build_stats_html(audio_path, elapsed)


def update_char_info(text: str) -> str:
    text = text or ""
    words = len(text.split()) if text.strip() else 0
    est = round(words / 2.5) if words else 0
    return (
        f'<div class="vc-meta">'
        f'<span>{len(text)} chars</span>'
        f'<span class="dot">&middot;</span>'
        f'<span>~{est}s speech</span>'
        f'</div>'
    )


def show_quality(value) -> str:
    return QUALITY_IDLE if value is None else QUALITY_READY


# ── HTML fragments ─────────────────────────────────────────────────────────────
LANG_CHOICES = [(name, code) for name, code in LANGUAGES]

QUALITY_IDLE = """
<div class="vc-status-line">
  <span class="vc-dot idle"></span>
  <span>No voice sample loaded</span>
</div>
"""

QUALITY_READY = """
<div class="vc-status-line">
  <span class="vc-dot ready"></span>
  <span>Sample ready &mdash; good quality</span>
</div>
"""

# ── Result header HTML ──────────────────────────────────────────────────────────
RESULT_HEADER = """
<div class="vc-result-header">
  <span class="vc-result-icon">&#10003;</span>
  <span>
    <div class="vc-result-title">Generated Audio</div>
    <div class="vc-result-sub">Ready to play &amp; download</div>
  </span>
</div>
"""

# ── CSS ────────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ════ TOKENS ════ */
:root {
  --bg:      #0d0d0f;
  --s1:      #141416;
  --s2:      #1c1c1f;
  --s3:      #242428;
  --border:  #2a2a2e;
  --border2: #333338;
  --accent:  #e8ff47;
  --teal:    #3dffc0;
  --red:     #ff4545;
  --text:    #f2f2f0;
  --t2:      #8a8a8f;
  --t3:      #4a4a50;
  --ff-ui:   'Inter', system-ui, sans-serif;
  --ff-mono: 'JetBrains Mono', 'Courier New', monospace;
}

/* ════ RESET ════ */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* ════ CRITICAL GRADIO OVERRIDES ════ */
.gradio-container, body {
  background: var(--bg) !important;
  font-family: var(--ff-ui) !important;
  color: var(--text) !important;
  min-height: 100vh;
  -webkit-font-smoothing: antialiased;
}
.gradio-container .contain { padding: 0 !important; }
.gradio-container > .main  { padding: 0 !important; }
footer.svelte-mpyp5e, .footer { display: none !important; }

#vc-left-panel .block,
#vc-right-panel .block,
#vc-results .block {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin: 0 !important;
}
#vc-left-panel .wrap,
#vc-left-panel .form,
#vc-left-panel .wrap-inner,
#vc-right-panel .wrap,
#vc-right-panel .form,
#vc-results .wrap { padding: 0 !important; }

#vc-left-panel label > span,
#vc-right-panel label > span { display: none !important; }

#vc-workspace > div { gap: 0 !important; }

#vc-left-panel > .wrap {
  display: flex !important;
  flex-direction: column !important;
  padding: 20px !important;
  height: 100% !important;
  gap: 0 !important;
}
#vc-right-panel > .wrap {
  display: flex !important;
  flex-direction: column !important;
  padding: 20px 22px !important;
  height: 100% !important;
  gap: 0 !important;
}
#vc-results > .wrap { padding: 0 !important; gap: 0 !important; }

#mic-audio-input audio { display: none !important; }
#mic-audio-input .waveform-container,
#mic-audio-input [class*="waveform"] {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
}

*:focus { outline: none; }

/* ════ ZONE 1 — NAV ════ */
.vc-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 48px;
  padding: 0 32px;
  background: var(--s1);
  border-bottom: 1px solid var(--border);
}
.vc-logo {
  font-family: var(--ff-ui);
  font-size: 14px;
  font-weight: 600;
  letter-spacing: -0.02em;
  color: var(--text);
}
.vc-logo .clone { color: var(--accent); }
.vc-nav-pills {
  display: flex;
  align-items: center;
  gap: 8px;
}
.vc-nav-pills span {
  font-family: var(--ff-mono);
  font-size: 10px;
  padding: 3px 8px;
  border-radius: 20px;
  border: 1px solid var(--border2);
}
.vc-nav-pills .pill-xtts {
  color: var(--teal);
  background: rgba(61,255,192,0.07);
  border-color: rgba(61,255,192,0.2);
}
.vc-nav-pills .pill-dim { color: var(--t3); }

/* ════ ZONE 2 — HERO ════ */
.vc-hero {
  position: relative;
  text-align: center;
  padding: 36px 24px 28px;
  background: var(--s1);
  overflow: hidden;
}
.vc-hero-tint {
  position: absolute;
  inset: 0;
  pointer-events: none;
  background: radial-gradient(ellipse 60% 80% at 50% -20%, rgba(232,255,71,0.04) 0%, transparent 70%);
}
.vc-hero-body { position: relative; z-index: 1; }

.vc-hero-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(232,255,71,0.06);
  border: 1px solid rgba(232,255,71,0.18);
  border-radius: 20px;
  padding: 4px 10px;
  font-family: var(--ff-mono);
  font-size: 10px;
  color: var(--t2);
  margin-bottom: 16px;
}
.vc-chip-dot {
  display: inline-block;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 6px var(--accent);
  flex-shrink: 0;
}

.vc-headline {
  font-family: var(--ff-ui);
  font-size: 26px;
  font-weight: 600;
  letter-spacing: -0.04em;
  line-height: 1.1;
  color: var(--text);
  margin-bottom: 10px;
}
.vc-headline .voice-word { color: var(--accent); }

.vc-subline {
  font-family: var(--ff-ui);
  font-size: 12px;
  font-weight: 400;
  color: var(--t2);
  max-width: 360px;
  margin: 0 auto 16px;
  line-height: 1.6;
}

.vc-hero-pills {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 20px;
}
.vc-hero-pills span {
  font-family: var(--ff-mono);
  font-size: 10px;
  color: var(--t3);
  border: 1px solid var(--border2);
  padding: 3px 8px;
  border-radius: 20px;
  background: rgba(255,255,255,0.015);
}

/* Wave — 9 bars */
.vc-wave {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 3px;
}
.vc-wave span {
  display: inline-block;
  width: 2.5px;
  border-radius: 2px;
  background: var(--accent);
  opacity: 0.3;
  animation: wv 1.4s ease-in-out infinite;
  transform-origin: bottom center;
}
.vc-wave span:nth-child(1) { height:  6px; animation-delay: 0.0s; }
.vc-wave span:nth-child(2) { height: 11px; animation-delay: 0.1s; }
.vc-wave span:nth-child(3) { height: 18px; animation-delay: 0.2s; }
.vc-wave span:nth-child(4) { height: 24px; animation-delay: 0.3s; }
.vc-wave span:nth-child(5) { height: 28px; animation-delay: 0.4s; }
.vc-wave span:nth-child(6) { height: 24px; animation-delay: 0.5s; }
.vc-wave span:nth-child(7) { height: 18px; animation-delay: 0.6s; }
.vc-wave span:nth-child(8) { height: 11px; animation-delay: 0.7s; }
.vc-wave span:nth-child(9) { height:  6px; animation-delay: 0.8s; }
@keyframes wv {
  0%, 100% { transform: scaleY(0.3);  opacity: 0.15; }
  50%       { transform: scaleY(1.0);  opacity: 0.60; }
}

/* ════ APP WRAPPER ════ */
#vc-app-wrapper {
  max-width: 1180px;
  margin: 0 auto;
  padding: 0 0 80px;
}
#vc-app-wrapper > .wrap { padding: 0 !important; }

/* ════ ZONE 3 — WORKSPACE ════ */
#vc-workspace {
  display: flex !important;
  min-height: 440px !important;
  gap: 0 !important;
  align-items: stretch !important;
  border-top: 1px solid var(--border) !important;
  border-bottom: 1px solid var(--border) !important;
  overflow: visible !important;
  margin-bottom: 0 !important;
}

/* ── LEFT PANEL ── */
#vc-left-panel {
  border-right: 1px solid var(--border) !important;
  flex: 0 0 36% !important;
  max-width: 36% !important;
}

/* Section headers */
.vc-sec-head {
  display: flex;
  align-items: center;
  gap: 7px;
  margin-bottom: 10px;
}
.vc-sec-title {
  font-family: var(--ff-mono);
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--t3);
}
.vc-sec-hint {
  font-family: var(--ff-mono);
  font-size: 10px;
  color: var(--t3);
  margin-left: auto;
}
.vc-step-num {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 17px;
  height: 17px;
  min-width: 17px;
  border-radius: 50%;
  font-family: var(--ff-mono);
  font-size: 9px;
  font-weight: 600;
  background: rgba(232,255,71,0.1);
  border: 1px solid rgba(232,255,71,0.22);
  color: var(--accent);
}
.vc-step-num.step-rec {
  background: rgba(255,69,69,0.1);
  border-color: rgba(255,69,69,0.25);
  color: var(--red);
}
.vc-step-num.step-2 {
  background: rgba(61,255,192,0.08);
  border-color: rgba(61,255,192,0.2);
  color: var(--teal);
}

/* Right-panel section header same pattern */
.vc-right-header {
  display: flex;
  align-items: center;
  gap: 7px;
  margin-bottom: 10px;
}

/* Upload zone */
#file-upload-input { margin-bottom: 0 !important; }
#file-upload-input .upload-container,
#file-upload-input [data-testid="upload-zone"] {
  background: rgba(255,255,255,0.015) !important;
  border: 1.5px dashed var(--border2) !important;
  border-radius: 10px !important;
  padding: 20px 14px !important;
  text-align: center !important;
  cursor: pointer !important;
  transition: border-color 0.2s, background 0.2s !important;
}
#file-upload-input .upload-container:hover,
#file-upload-input [data-testid="upload-zone"]:hover {
  border-color: rgba(232,255,71,0.22) !important;
  background: rgba(232,255,71,0.04) !important;
}
#file-upload-input .upload-container span,
#file-upload-input [data-testid="upload-zone"] span {
  font-family: var(--ff-mono) !important;
  font-size: 10px !important;
  color: var(--t3) !important;
}

/* Isolate checkbox */
#isolate-chk { margin-bottom: 14px !important; }
#isolate-chk label {
  display: flex !important;
  flex-direction: row !important;
  align-items: center !important;
  gap: 7px !important;
  padding: 8px 10px !important;
  border-radius: 8px !important;
  border: 1px solid var(--border) !important;
  background: var(--s2) !important;
  cursor: pointer !important;
}
#isolate-chk label span {
  display: inline !important;
  font-family: var(--ff-ui) !important;
  font-size: 11px !important;
  font-weight: 400 !important;
  color: var(--t2) !important;
  text-transform: none !important;
  letter-spacing: 0 !important;
}
#isolate-chk .recommended-tag {
  font-family: var(--ff-mono);
  font-size: 10px;
  color: var(--t3);
  margin-left: auto;
}
input[type="checkbox"] { accent-color: var(--accent); }

/* OR separator */
.vc-or-sep {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 2px 0 16px;
  font-family: var(--ff-mono);
  font-size: 10px;
  color: var(--t3);
  user-select: none;
}
.vc-or-sep::before, .vc-or-sep::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}

/* Mic container */
#mic-audio-input {
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 12px !important;
  padding: 12px 0 !important;
}
#mic-audio-input > .wrap,
#mic-audio-input .block {
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  width: 100% !important;
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  margin: 0 !important;
}

/* ── RECORD BUTTON (confirmed class selectors) ── */
#mic-audio-input button.record.record-button {
  width: 64px !important;
  height: 64px !important;
  border-radius: 50% !important;
  background: rgba(255,69,69,0.1) !important;
  border: 2px solid rgba(255,69,69,0.3) !important;
  color: var(--red) !important;
  margin: 0 auto !important;
  cursor: pointer !important;
  position: relative !important;
  font-size: 0 !important;
  flex-shrink: 0 !important;
}
#mic-audio-input button.record.record-button::after {
  content: '' !important;
  position: absolute !important;
  top: 50% !important;
  left: 50% !important;
  transform: translate(-50%, -50%) !important;
  width: 22px !important;
  height: 22px !important;
  border-radius: 50% !important;
  background: var(--red) !important;
}
#mic-audio-input button.record.record-button::before {
  content: '' !important;
  position: absolute !important;
  inset: -6px !important;
  border-radius: 50% !important;
  border: 1.5px solid rgba(255,69,69,0.15) !important;
  animation: rec-pulse 2.2s ease-in-out infinite !important;
}
@keyframes rec-pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%       { opacity: 0; transform: scale(1.35); }
}

#mic-audio-input button.pause-button {
  width: 64px !important;
  height: 64px !important;
  border-radius: 50% !important;
  background: rgba(255,69,69,0.15) !important;
  border: 2px solid rgba(255,69,69,0.4) !important;
  margin: 0 auto !important;
  cursor: pointer !important;
}
#mic-audio-input button.pause-button svg {
  fill: var(--red) !important;
  stroke: var(--red) !important;
  width: 22px !important;
  height: 22px !important;
}

.vc-mic-hint {
  font-family: var(--ff-mono);
  font-size: 10px;
  color: var(--t3);
  text-align: center;
  line-height: 1.5;
}

/* Status line */
.vc-status-line {
  display: flex;
  align-items: center;
  gap: 8px;
  padding-top: 14px;
  margin-top: auto;
  border-top: 1px solid var(--border);
  font-family: var(--ff-ui);
  font-size: 11px;
  color: var(--t3);
}
.vc-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
}
.vc-dot.idle  { background: var(--t3); }
.vc-dot.ready {
  background: var(--teal);
  box-shadow: 0 0 6px rgba(61,255,192,0.5);
  animation: blink 2s ease-in-out infinite;
}
@keyframes blink {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.3; }
}

/* Clone status textbox */
.vc-clone-status {
  border: none !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin: 0 !important;
  font-family: var(--ff-mono) !important;
  font-size: 10px !important;
  color: var(--t3) !important;
  resize: none !important;
  outline: none !important;
  min-height: unset !important;
  line-height: 1.55 !important;
}
.vc-clone-status:focus { box-shadow: none !important; }

/* ── RIGHT PANEL ── */
#vc-right-panel { flex: 1 !important; }

#script-text-input { margin-bottom: 14px !important; }
#script-text-input > .wrap,
#script-text-input label,
#script-text-input .wrap-inner {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}
#script-text-input textarea {
  flex: 1 !important;
  min-height: 220px !important;
  max-height: 320px !important;
  resize: vertical !important;
  background: rgba(5,5,8,0.6) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 14px 16px !important;
  font-family: var(--ff-ui) !important;
  font-size: 13px !important;
  line-height: 1.7 !important;
  color: var(--text) !important;
  caret-color: var(--accent) !important;
  outline: none !important;
  width: 100% !important;
}
#script-text-input textarea::placeholder { color: var(--t3) !important; }
#script-text-input textarea:focus {
  border-color: rgba(232,255,71,0.3) !important;
  box-shadow: 0 0 0 3px rgba(232,255,71,0.06) !important;
  outline: none !important;
}
#script-text-input label > span { display: none !important; }

/* Script footer */
#vc-script-footer {
  display: flex !important;
  flex-direction: row !important;
  gap: 10px !important;
  align-items: center !important;
  padding-top: 12px !important;
  border-top: 1px solid var(--border) !important;
  flex-wrap: nowrap !important;
}
#char-info   { flex: 1 1 auto !important; min-width: 0 !important; }
#lang-select { flex: 0 0 auto !important; min-width: 120px !important; }
#gen-btn     { flex: 0 0 auto !important; }

.vc-meta {
  display: flex;
  align-items: center;
  gap: 6px;
  font-family: var(--ff-mono);
  font-size: 10px;
  color: var(--t3);
}
.vc-meta .dot { color: var(--t2); font-weight: 500; opacity: 0.6; }
.vc-meta span { color: var(--t2); font-weight: 500; }

/* Language dropdown */
#lang-select label > span { display: none !important; }
#lang-select select,
#lang-select .wrap-inner,
#lang-select [class*="dropdown"] {
  background: var(--s2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 8px !important;
  padding: 7px 10px !important;
  font-family: var(--ff-ui) !important;
  font-size: 11px !important;
  color: var(--t2) !important;
  cursor: pointer !important;
}

/* Generate button */
#gen-btn button {
  background: var(--accent) !important;
  color: #0a0a0a !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 9px 22px !important;
  font-family: var(--ff-ui) !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  letter-spacing: -0.01em !important;
  white-space: nowrap !important;
  cursor: pointer !important;
  transition: opacity 0.15s, transform 0.15s !important;
  position: relative !important;
  overflow: hidden !important;
}
#gen-btn button:hover  { opacity: 0.9 !important; transform: translateY(-1px) !important; }
#gen-btn button:active { transform: translateY(0) !important; }

#gen-btn button.vc-btn-loading {
  opacity: 0.5 !important;
  pointer-events: none !important;
}
#gen-btn button.vc-btn-loading::after {
  content: '' !important;
  position: absolute !important;
  right: 14px !important;
  top: 50% !important;
  transform: translateY(-50%) !important;
  width: 13px !important;
  height: 13px !important;
  border: 2px solid rgba(0,0,0,0.2) !important;
  border-top-color: #0a0a0a !important;
  border-radius: 50% !important;
  animation: vc-spin 0.7s linear infinite !important;
}
@keyframes vc-spin { to { transform: translateY(-50%) rotate(360deg); } }

/* ════ ZONE 4 — RESULTS ════ */
#vc-results {
  background: var(--s1) !important;
  border-top: 1px solid var(--border) !important;
  padding: 16px 20px !important;
  gap: 0 !important;
}

.vc-result-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
}
.vc-result-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 7px;
  background: rgba(61,255,192,0.07);
  border: 1px solid rgba(61,255,192,0.2);
  color: var(--teal);
  font-size: 14px;
  flex-shrink: 0;
}
.vc-result-title {
  font-family: var(--ff-ui);
  font-size: 12px;
  font-weight: 500;
  color: var(--text);
}
.vc-result-sub {
  font-family: var(--ff-mono);
  font-size: 10px;
  color: var(--t3);
}

#audio-output audio { width: 100% !important; }
#audio-output .waveform-container {
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  background: rgba(5,5,8,0.5) !important;
  padding: 12px 14px !important;
}

.vc-gen-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 10px;
  font-family: var(--ff-mono);
  font-size: 10px;
  color: var(--t3);
}

/* ════ ZONE 5 — FOOTER ════ */
.vc-footer {
  text-align: center;
  padding: 24px;
  border-top: 1px solid var(--border);
  font-family: var(--ff-mono);
  font-size: 10px;
  color: var(--t3);
}
.vc-footer a { color: var(--t2); text-decoration: none; }
.vc-footer a:hover { color: var(--text); }

/* ════ GLOBAL INPUT ════ */
input[type="text"] {
  background: var(--bg) !important;
  color: var(--text) !important;
}

/* ════ MOBILE ════ */
@media (max-width: 760px) {
  #vc-workspace {
    flex-direction: column !important;
    min-height: unset !important;
  }
  #vc-left-panel {
    flex: unset !important;
    max-width: 100% !important;
    border-right: none !important;
    border-bottom: 1px solid var(--border) !important;
  }
  #vc-script-footer { flex-wrap: wrap !important; }
  #char-info { width: 100% !important; }
  #lang-select { flex: 1 1 auto !important; }
}
"""

# ── JavaScript ─────────────────────────────────────────────────────────────────
JS = """
function() {
  function setup() {
    var wrap = document.getElementById('gen-btn');
    if (!wrap) { setTimeout(setup, 500); return; }
    var btn = wrap.querySelector('button') || wrap;
    btn.addEventListener('click', function() {
      btn.classList.add('vc-btn-loading');
    });
    var out = document.getElementById('audio-output');
    if (out) {
      new MutationObserver(function() {
        btn.classList.remove('vc-btn-loading');
      }).observe(out, { childList: true, subtree: true });
    }
  }
  setup();
}
"""

# ── Gradio layout ──────────────────────────────────────────────────────────────
with gr.Blocks(
    title="VoiceClone AI — Free AI Voice Cloning",
    css=CSS,
    js=JS,
    theme=gr.themes.Base(),
) as demo:

    gr.HTML("""
    <div class="vc-header">
      <span class="vc-logo">Voice<span class="clone">Clone</span> AI</span>
      <div class="vc-nav-pills">
        <span class="pill-xtts">XTTS v2</span>
        <span class="pill-dim">GPU</span>
        <span class="pill-dim">v1.0</span>
      </div>
    </div>
    """)

    gr.HTML("""
    <div class="vc-hero">
      <div class="vc-hero-tint"></div>
      <div class="vc-hero-body">
        <div class="vc-hero-chip">
          <span class="vc-chip-dot"></span>AI Voice Cloning
        </div>
        <h1 class="vc-headline">Clone Any <span class="voice-word">Voice</span><br>in Seconds</h1>
        <p class="vc-subline">Upload a voice sample, type your script, and hear it spoken back in your cloned voice &mdash; instantly.</p>
        <div class="vc-hero-pills">
          <span>16 Languages</span>
          <span>Runs Locally</span>
          <span>GPU Accelerated</span>
          <span>No Account Needed</span>
        </div>
        <div class="vc-wave">
          <span></span><span></span><span></span><span></span><span></span>
          <span></span><span></span><span></span><span></span>
        </div>
      </div>
    </div>
    """)

    with gr.Column(elem_id="vc-app-wrapper"):

        # ── Single workspace container ────────────────────────────────────────
        with gr.Row(elem_id="vc-workspace"):

            # ── LEFT PANEL (30%) ──────────────────────────────────────────────
            with gr.Column(scale=3, min_width=260, elem_id="vc-left-panel"):

                gr.HTML("""
                <div class="vc-sec-head">
                  <span class="vc-step-num">1</span>
                  <span class="vc-sec-title">Upload Audio</span>
                  <span class="vc-sec-hint">MP4 or WAV</span>
                </div>
                """)

                # Upload zone
                file_upload = gr.File(
                    show_label=False,
                    file_types=[".mp4", ".wav"],
                    elem_id="file-upload-input",
                )

                if HAS_DEMUCS:
                    isolate_chk = gr.Checkbox(
                        label="Isolate vocals",
                        value=False,
                        elem_id="isolate-chk",
                    )
                else:
                    isolate_chk = gr.Checkbox(
                        label="Isolate vocals",
                        value=False,
                        interactive=False,
                        elem_id="isolate-chk",
                    )

                # "or" divider
                gr.HTML('<div class="vc-or-sep"><span>or</span></div>')

                # Record section — button centered, red
                gr.HTML("""
                <div class="vc-sec-head">
                  <span class="vc-step-num step-rec">&#9679;</span>
                  <span class="vc-sec-title">Record Voice</span>
                  <span class="vc-sec-hint">6&ndash;30 sec</span>
                </div>
                """)
                mic_audio = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    show_label=False,
                    elem_id="mic-audio-input",
                    waveform_options={
                        "waveform_color": "#3dffc0",
                        "waveform_progress_color": "#e8ff47",
                    },
                )

                gr.HTML('<p class="vc-mic-hint">6&ndash;30 sec for best quality</p>')

                # Voice quality indicator
                quality_card = gr.HTML(QUALITY_IDLE, elem_id="quality-card")

                # Clone status
                status_box = gr.Textbox(
                    show_label=False,
                    interactive=False,
                    placeholder="Ready to generate",
                    elem_classes="vc-clone-status",
                    elem_id="status-box",
                )

            # ── RIGHT PANEL (65%) ─────────────────────────────────────────────
            with gr.Column(scale=7, elem_id="vc-right-panel"):

                gr.HTML("""
                <div class="vc-right-header">
                  <span class="vc-step-num step-2">2</span>
                  <span class="vc-sec-title">Your Script</span>
                  <span class="vc-sec-hint">Type what to say</span>
                </div>
                """)

                script_text = gr.Textbox(
                    placeholder="Type what you want spoken in your voice...",
                    lines=10,
                    max_lines=30,
                    show_label=False,
                    elem_id="script-text-input",
                )

                # Footer bar: meta | language | generate
                with gr.Row(elem_id="vc-script-footer"):
                    char_info = gr.HTML(
                        '<div class="vc-meta">'
                        '<span>0 chars</span>'
                        '<span class="dot">&middot;</span>'
                        '<span>~0s speech</span>'
                        '</div>',
                        elem_id="char-info",
                    )
                    lang_select = gr.Dropdown(
                        choices=LANG_CHOICES,
                        value="en",
                        show_label=False,
                        interactive=True,
                        elem_id="lang-select",
                    )
                    gen_btn = gr.Button(
                        "Generate",
                        elem_id="gen-btn",
                    )

        # ── Results (below workspace) ─────────────────────────────────────────
        with gr.Column(elem_id="vc-results"):
            gr.HTML(RESULT_HEADER)
            audio_out = gr.Audio(
                type="filepath",
                show_label=False,
                elem_id="audio-output",
            )
            stats_html = gr.HTML("", elem_id="gen-stats")

    gr.HTML("""
    <div class="vc-footer">
      <a href="https://github.com/coqui-ai/TTS" target="_blank">Coqui XTTS v2</a>
      &nbsp;&middot;&nbsp; Runs locally &mdash; nothing is uploaded
    </div>
    """)

    # ── Event bindings ─────────────────────────────────────────────────────────
    script_text.change(fn=update_char_info, inputs=[script_text], outputs=[char_info])
    mic_audio.change(fn=show_quality, inputs=[mic_audio], outputs=[quality_card])
    file_upload.change(fn=show_quality, inputs=[file_upload], outputs=[quality_card])

    gen_btn.click(
        fn=run_unified,
        inputs=[mic_audio, file_upload, isolate_chk, script_text, lang_select],
        outputs=[audio_out, status_box, stats_html],
    )

demo.launch()
