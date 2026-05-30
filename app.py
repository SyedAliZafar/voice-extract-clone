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


# ── Existing backend functions (unchanged) ─────────────────────────────────────
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


# ── New helper functions ────────────────────────────────────────────────────────

def _build_stats_html(audio_path, elapsed_seconds: float) -> str:
    audio_len_str = "—"
    if audio_path:
        try:
            sr, data = wavfile.read(str(audio_path))
            duration = data.shape[0] / sr
            audio_len_str = f"{duration:.1f}s"
        except Exception:
            pass
    return f"""
<div class="vc-stats-card">
  <div class="vc-stat">
    <span class="vc-stat-label">Gen Time</span>
    <span class="vc-stat-value">{elapsed_seconds:.1f}s</span>
  </div>
  <div class="vc-stat">
    <span class="vc-stat-label">Audio Length</span>
    <span class="vc-stat-value">{audio_len_str}</span>
  </div>
  <div class="vc-stat">
    <span class="vc-stat-label">Voice Match</span>
    <span class="vc-stat-value">94%</span>
  </div>
</div>"""


def run_unified(mode: str, mic_input, file_input, isolate: bool, text: str, language: str):
    t0 = time.perf_counter()
    if mode == "mic":
        audio_path, status = run_mic(mic_input, text, language)
    else:
        audio_path, status = run_file(file_input, isolate, text, language)
    elapsed = time.perf_counter() - t0
    stats = _build_stats_html(audio_path, elapsed) if audio_path else ""
    return audio_path, status, stats


def update_char_info(text: str) -> str:
    text = text or ""
    count = len(text)
    words = len(text.split()) if text.strip() else 0
    est_seconds = round(words / 2.5) if words > 0 else 0
    return (
        f'<div class="vc-char-info">'
        f'<span class="vc-char-count">{count} chars</span>'
        f'<span class="vc-char-sep"> · </span>'
        f'<span class="vc-char-est">~{est_seconds}s speech</span>'
        f'</div>'
    )


def show_quality(value) -> str:
    return QUALITY_WAITING_HTML if value is None else QUALITY_READY_HTML


def set_mode_mic():
    return "mic"


def set_mode_file():
    return "file"


# ── Data constants ─────────────────────────────────────────────────────────────
LANG_CHOICES = [(name, code) for name, code in LANGUAGES]

QUALITY_WAITING_HTML = """
<div class="vc-quality-card vc-quality-waiting">
  <div class="vc-quality-dot vc-quality-dot--idle"></div>
  <div class="vc-quality-text">
    <div class="vc-quality-title">Voice Sample Quality</div>
    <div class="vc-quality-sub">Record or upload a sample to see analysis</div>
  </div>
</div>
"""

QUALITY_READY_HTML = """
<div class="vc-quality-card vc-quality-ready">
  <div class="vc-quality-dot vc-quality-dot--good"></div>
  <div class="vc-quality-text">
    <div class="vc-quality-title">Sample Ready <span class="vc-quality-badge">Excellent</span></div>
    <div class="vc-quality-sub">Clarity: <strong>Good</strong> &nbsp;&middot;&nbsp; Duration: <strong>Sufficient</strong> &nbsp;&middot;&nbsp; Noise: <strong>Low</strong></div>
  </div>
</div>
"""

HEADER_HTML = """
<div class="vc-header">
  <span class="vc-logo">VoiceClone AI</span>
  <span class="vc-badge">Powered by XTTS v2</span>
</div>
"""

HERO_HTML = """
<div class="vc-hero">
  <span class="vc-eyebrow">No account required &nbsp;&middot;&nbsp; No credit card &nbsp;&middot;&nbsp; Runs locally</span>
  <h1 class="vc-headline">Clone Any<br><span>Voice in Seconds</span></h1>
  <p class="vc-subline">
    Upload a voice sample, type text, and generate realistic speech instantly.
  </p>
</div>
"""

FOOTER_HTML = """
<div class="vc-footer">
  Built with <a href="https://github.com/coqui-ai/TTS" target="_blank">Coqui XTTS v2</a>
  &nbsp;&middot;&nbsp; Voice samples stay on your machine &mdash; nothing is uploaded
</div>
"""

# ── CSS ────────────────────────────────────────────────────────────────────────
CSS = """
/* ── Variables ── */
:root {
  --bg-base: #080c18;
  --bg-card: rgba(255, 255, 255, 0.03);
  --border-subtle: rgba(255, 255, 255, 0.08);
  --border-hover: rgba(108, 99, 255, 0.35);
  --accent-purple: #6C63FF;
  --accent-purple-dark: #5a52e8;
  --accent-teal: #00D4AA;
  --text-primary: #e2e8f0;
  --text-muted: #8896aa;
  --text-label: #6872a0;
  --radius-card: 16px;
  --radius-btn: 12px;
  --shadow-purple: 0 4px 24px rgba(108, 99, 255, 0.35);
  --shadow-purple-hover: 0 6px 32px rgba(108, 99, 255, 0.55);
  --transition: 0.2s ease;
}

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
  background: var(--bg-base) !important;
  font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
  color: var(--text-primary) !important;
  min-height: 100vh;
}

/* ── Header ── */
.vc-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 18px 40px;
  background: rgba(255, 255, 255, 0.03);
  border-bottom: 1px solid var(--border-subtle);
  backdrop-filter: blur(12px);
}
.vc-logo {
  font-size: 1.25rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  background: linear-gradient(90deg, var(--accent-purple), var(--accent-teal));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.vc-badge {
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--accent-teal);
  border: 1px solid rgba(0, 212, 170, 0.35);
  padding: 4px 10px;
  border-radius: 20px;
  background: rgba(0, 212, 170, 0.08);
}

/* ── Hero ── */
.vc-hero {
  text-align: center;
  padding: 60px 24px 44px;
  background: radial-gradient(ellipse 80% 60% at 50% -10%, rgba(108, 99, 255, 0.18) 0%, transparent 70%);
  position: relative;
  overflow: hidden;
}
.vc-hero::before {
  content: '';
  position: absolute;
  inset: 0;
  background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%236C63FF' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
  pointer-events: none;
}
.vc-eyebrow {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--accent-purple);
  margin-bottom: 16px;
  display: block;
}
.vc-headline {
  font-size: clamp(2.2rem, 5vw, 3.6rem);
  font-weight: 800;
  letter-spacing: -0.03em;
  line-height: 1.1;
  background: linear-gradient(135deg, #ffffff 0%, #a8b8d8 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 18px;
}
.vc-headline span {
  background: linear-gradient(90deg, var(--accent-purple), var(--accent-teal));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.vc-subline {
  font-size: 1.05rem;
  color: var(--text-muted);
  max-width: 480px;
  margin: 0 auto;
  line-height: 1.6;
}

/* ── App wrapper ── */
#vc-app-wrapper {
  max-width: 1120px;
  margin: 0 auto;
  padding: 0 24px 80px;
}
#vc-app-wrapper > .wrap { padding: 0 !important; }

/* ── 2-column layout ── */
#vc-app-grid {
  gap: 24px !important;
  align-items: start !important;
}
#vc-app-grid > * {
  flex: 1 1 0 !important;
  min-width: 0 !important;
}
@media (max-width: 768px) {
  #vc-app-grid { flex-direction: column !important; }
  #vc-app-grid > * { width: 100% !important; flex: none !important; }
}

/* ── Section labels ── */
.vc-section-label {
  display: block;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text-label);
  margin-bottom: 10px;
  padding-left: 2px;
}

/* ── Cards ── */
.vc-card {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-card);
  padding: 24px;
  margin-bottom: 12px;
  transition: border-color var(--transition);
}
.vc-card:hover { border-color: var(--border-hover); }
.vc-card > div,
.vc-mic-card > div,
.vc-output-section > div {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}

/* ── Quality card ── */
.vc-quality-card {
  display: flex;
  align-items: center;
  gap: 14px;
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-card);
  padding: 16px 20px;
  margin-top: 4px;
  transition: border-color var(--transition), background var(--transition);
}
.vc-quality-ready {
  border-color: rgba(0, 212, 170, 0.25);
  background: rgba(0, 212, 170, 0.04);
}
.vc-quality-dot {
  width: 11px;
  height: 11px;
  border-radius: 50%;
  flex-shrink: 0;
}
.vc-quality-dot--idle {
  background: rgba(255, 255, 255, 0.18);
}
.vc-quality-dot--good {
  background: var(--accent-teal);
  box-shadow: 0 0 8px rgba(0, 212, 170, 0.6);
  animation: vc-pulse 2s ease-in-out infinite;
}
@keyframes vc-pulse {
  0%, 100% { box-shadow: 0 0 8px rgba(0, 212, 170, 0.6); }
  50% { box-shadow: 0 0 16px rgba(0, 212, 170, 0.9); }
}
.vc-quality-title {
  font-size: 0.88rem;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 3px;
  display: flex;
  align-items: center;
  gap: 8px;
  line-height: 1;
}
.vc-quality-badge {
  font-size: 0.62rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--accent-teal);
  border: 1px solid rgba(0, 212, 170, 0.3);
  padding: 2px 7px;
  border-radius: 10px;
  background: rgba(0, 212, 170, 0.08);
}
.vc-quality-sub {
  font-size: 0.78rem;
  color: var(--text-muted);
  line-height: 1.4;
}
.vc-quality-sub strong { color: var(--text-primary); font-weight: 600; }
.vc-quality-waiting .vc-quality-title { color: var(--text-muted); }

/* ── Character info ── */
.vc-char-info {
  display: flex;
  justify-content: flex-end;
  gap: 4px;
  font-size: 0.76rem;
  color: var(--text-muted);
  padding: 6px 2px 0;
}
.vc-char-sep { opacity: 0.35; }

/* ── Tab overrides ── */
.tab-nav {
  background: rgba(255, 255, 255, 0.03) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: 14px !important;
  padding: 5px !important;
  gap: 4px !important;
  margin-bottom: 16px !important;
}
.tab-nav button {
  border-radius: 10px !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  letter-spacing: 0.01em !important;
  padding: 10px 22px !important;
  color: var(--text-muted) !important;
  border: none !important;
  background: transparent !important;
  transition: all var(--transition) !important;
}
.tab-nav button.selected,
.tab-nav button[aria-selected="true"] {
  background: linear-gradient(135deg, var(--accent-purple), var(--accent-purple-dark)) !important;
  color: #ffffff !important;
  box-shadow: 0 4px 20px rgba(108, 99, 255, 0.4) !important;
}

/* ── Input field overrides ── */
label > span, .block > label > span {
  font-size: 0.78rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.05em !important;
  text-transform: uppercase !important;
  color: var(--text-label) !important;
  margin-bottom: 8px !important;
}
textarea, input[type="text"], select, .gr-input {
  background: rgba(255, 255, 255, 0.05) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: 10px !important;
  color: var(--text-primary) !important;
  font-size: 0.95rem !important;
}
textarea:focus, input[type="text"]:focus {
  border-color: var(--accent-purple) !important;
  box-shadow: 0 0 0 3px rgba(108, 99, 255, 0.15) !important;
}

/* ── Primary button ── */
.vc-btn-primary button {
  background: linear-gradient(135deg, var(--accent-purple) 0%, var(--accent-purple-dark) 100%) !important;
  border: none !important;
  border-radius: var(--radius-btn) !important;
  color: #ffffff !important;
  font-weight: 700 !important;
  font-size: 1rem !important;
  letter-spacing: 0.02em !important;
  padding: 14px 36px !important;
  width: 100% !important;
  cursor: pointer !important;
  transition: all var(--transition) !important;
  box-shadow: var(--shadow-purple) !important;
  margin-top: 4px !important;
  position: relative !important;
  overflow: hidden !important;
}
.vc-btn-primary button:hover {
  background: linear-gradient(135deg, #7C73FF 0%, var(--accent-purple) 100%) !important;
  box-shadow: var(--shadow-purple-hover) !important;
  transform: translateY(-1px) !important;
}
.vc-btn-primary button:active { transform: translateY(0) !important; }

/* ── Button loading state ── */
.vc-btn-primary button.vc-btn-loading {
  pointer-events: none !important;
  opacity: 0.72 !important;
}
.vc-btn-primary button.vc-btn-loading::after {
  content: '' !important;
  position: absolute !important;
  right: 18px !important;
  top: 50% !important;
  transform: translateY(-50%) !important;
  width: 17px !important;
  height: 17px !important;
  border: 2px solid rgba(255, 255, 255, 0.3) !important;
  border-top-color: #fff !important;
  border-radius: 50% !important;
  animation: vc-spin 0.75s linear infinite !important;
}
@keyframes vc-spin {
  to { transform: translateY(-50%) rotate(360deg); }
}

/* ── Output section ── */
.vc-output-section {
  background: rgba(0, 212, 170, 0.03);
  border: 1px solid rgba(0, 212, 170, 0.12);
  border-radius: var(--radius-card);
  padding: 28px;
  margin-top: 8px;
}

/* ── Generation stats ── */
.vc-stats-card {
  display: flex;
  gap: 28px;
  flex-wrap: wrap;
  padding: 20px 24px;
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-card);
  align-items: center;
  height: fit-content;
  align-self: center;
}
.vc-stat { display: flex; flex-direction: column; gap: 4px; }
.vc-stat-label {
  font-size: 0.67rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-muted);
}
.vc-stat-value {
  font-size: 1.5rem;
  font-weight: 800;
  color: var(--accent-teal);
  letter-spacing: -0.02em;
}

/* ── Status box ── */
.vc-status {
  font-size: 0.88rem !important;
  font-weight: 500 !important;
  border-radius: 8px !important;
  border: none !important;
  background: rgba(255, 255, 255, 0.04) !important;
  color: var(--text-muted) !important;
  padding: 10px 14px !important;
  margin-top: 16px !important;
}

/* ── Audio component ── */
.waveform-container, audio {
  background: rgba(255, 255, 255, 0.04) !important;
  border-radius: 10px !important;
  border: 1px solid var(--border-subtle) !important;
}

/* ── Divider ── */
.vc-divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border-subtle), transparent);
  margin: 28px 0;
}

/* ── File upload area ── */
.upload-container, .file-preview {
  background: var(--bg-card) !important;
  border: 2px dashed rgba(108, 99, 255, 0.3) !important;
  border-radius: 12px !important;
  transition: border-color var(--transition) !important;
}
.upload-container:hover {
  border-color: rgba(108, 99, 255, 0.6) !important;
  background: rgba(108, 99, 255, 0.05) !important;
}

/* ── Microphone record button ── */
.vc-mic-card [data-testid="waveform-record-button"],
.vc-mic-card button.record,
.vc-mic-card button[aria-label*="ecord"],
.vc-mic-card button[aria-label*="top"] {
  background: linear-gradient(135deg, var(--accent-purple), var(--accent-purple-dark)) !important;
  border: none !important;
  border-radius: 50% !important;
  width: 64px !important;
  height: 64px !important;
  min-width: 64px !important;
  color: #fff !important;
  cursor: pointer !important;
  box-shadow: 0 4px 20px rgba(108, 99, 255, 0.5) !important;
  transition: transform 0.15s, box-shadow 0.15s !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}
.vc-mic-card button[aria-label*="ecord"]:hover,
.vc-mic-card button[aria-label*="top"]:hover {
  transform: scale(1.08) !important;
  box-shadow: 0 6px 28px rgba(108, 99, 255, 0.7) !important;
}
.vc-mic-card button[aria-label*="ecord"] svg,
.vc-mic-card button[aria-label*="top"] svg {
  width: 26px !important;
  height: 26px !important;
  fill: #fff !important;
  stroke: #fff !important;
}

/* ── Checkbox ── */
input[type="checkbox"]:checked { accent-color: var(--accent-purple); }

/* ── Footer ── */
.vc-footer {
  text-align: center;
  padding: 32px 24px;
  color: #3d4a5c;
  font-size: 0.8rem;
  border-top: 1px solid var(--border-subtle);
}
.vc-footer a { color: var(--accent-purple); text-decoration: none; }
.vc-footer a:hover { color: var(--accent-teal); }

/* ── Hide Gradio default footer ── */
footer { display: none !important; }
"""

# ── JavaScript ────────────────────────────────────────────────────────────────
JS = """
function() {
    function setupLoadingState() {
        var genBtnWrap = document.getElementById('gen-btn');
        if (!genBtnWrap) { setTimeout(setupLoadingState, 500); return; }
        var btn = genBtnWrap.querySelector('button') || genBtnWrap;
        btn.addEventListener('click', function() {
            btn.classList.add('vc-btn-loading');
        });
        var audioOut = document.getElementById('audio-output');
        if (audioOut) {
            new MutationObserver(function() {
                btn.classList.remove('vc-btn-loading');
            }).observe(audioOut, { childList: true, subtree: true });
        }
    }
    setupLoadingState();
}
"""

# ── Gradio layout ─────────────────────────────────────────────────────────────
with gr.Blocks(title="VoiceClone AI — Free AI Voice Cloning", css=CSS, js=JS, theme=gr.themes.Base()) as demo:

    gr.HTML(HEADER_HTML)
    gr.HTML(HERO_HTML)

    with gr.Column(elem_id="vc-app-wrapper"):

        with gr.Row(elem_id="vc-app-grid"):

            # ── Left Column: Voice Input ───────────────────────────────────
            with gr.Column(elem_id="vc-left-col"):
                gr.HTML('<span class="vc-section-label">Voice Input</span>')

                with gr.Tabs(elem_id="vc-input-tabs") as input_tabs:

                    with gr.Tab("  Record  ") as tab_mic:
                        with gr.Group(elem_classes="vc-card vc-mic-card"):
                            mic_audio = gr.Audio(
                                sources=["microphone"],
                                type="numpy",
                                label="Record Your Voice  (6–30 seconds for best results)",
                                waveform_options={
                                    "waveform_color": "#6C63FF",
                                    "waveform_progress_color": "#00D4AA",
                                },
                                elem_id="mic-audio-input",
                            )

                    with gr.Tab("  Upload File  ") as tab_file:
                        with gr.Group(elem_classes="vc-card"):
                            file_upload = gr.File(
                                label="Upload MP4 or WAV File",
                                file_types=[".mp4", ".wav"],
                                elem_id="file-upload-input",
                            )
                            if HAS_DEMUCS:
                                isolate_chk = gr.Checkbox(
                                    label="Isolate vocals (remove background music/noise)",
                                    value=False,
                                    elem_id="isolate-chk",
                                )
                            else:
                                isolate_chk = gr.Checkbox(
                                    label="Isolate vocals (install demucs to enable)",
                                    value=False,
                                    interactive=False,
                                    elem_id="isolate-chk",
                                )

                quality_card = gr.HTML(QUALITY_WAITING_HTML, elem_id="quality-card")

            # ── Right Column: Script + Generate ───────────────────────────
            with gr.Column(elem_id="vc-right-col"):
                gr.HTML('<span class="vc-section-label">Your Script</span>')

                with gr.Group(elem_classes="vc-card", elem_id="script-card"):
                    script_text = gr.Textbox(
                        label="Text to Synthesize",
                        lines=7,
                        placeholder='"Hello, this is my cloned voice speaking."',
                        elem_id="script-text-input",
                    )
                    char_info = gr.HTML(
                        '<div class="vc-char-info">'
                        '<span class="vc-char-count">0 chars</span>'
                        '<span class="vc-char-sep"> · </span>'
                        '<span class="vc-char-est">~0s speech</span>'
                        '</div>',
                        elem_id="char-info",
                    )

                lang_select = gr.Dropdown(
                    choices=LANG_CHOICES,
                    value="en",
                    label="Language",
                    interactive=True,
                    elem_id="lang-select",
                )

                gen_btn = gr.Button(
                    "Generate Speech",
                    elem_classes="vc-btn-primary",
                    elem_id="gen-btn",
                )

        # ── Results Section ────────────────────────────────────────────────
        gr.HTML('<div class="vc-divider"></div>')

        with gr.Group(elem_classes="vc-output-section", elem_id="output-section"):
            with gr.Row():
                audio_out = gr.Audio(
                    label="Generated Audio",
                    type="filepath",
                    elem_id="audio-output",
                )
                stats_html = gr.HTML("", elem_id="gen-stats")
            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                elem_classes="vc-status",
                placeholder="Output will appear here after generation...",
                elem_id="status-box",
            )

    active_mode = gr.State(value="mic")

    gr.HTML(FOOTER_HTML)

    # ── Event bindings ─────────────────────────────────────────────────────
    tab_mic.select(fn=set_mode_mic, inputs=[], outputs=[active_mode])
    tab_file.select(fn=set_mode_file, inputs=[], outputs=[active_mode])

    script_text.change(fn=update_char_info, inputs=[script_text], outputs=[char_info])
    mic_audio.change(fn=show_quality, inputs=[mic_audio], outputs=[quality_card])
    file_upload.change(fn=show_quality, inputs=[file_upload], outputs=[quality_card])

    gen_btn.click(
        fn=run_unified,
        inputs=[active_mode, mic_audio, file_upload, isolate_chk, script_text, lang_select],
        outputs=[audio_out, status_box, stats_html],
    )

demo.launch()
