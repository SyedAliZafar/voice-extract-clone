import shutil
import numpy as np
import scipy.io.wavfile as wavfile
import gradio as gr
from pathlib import Path

from scripts.extract import extract_audio, check_ffmpeg
from scripts.isolate import isolate_vocals, demucs_available
from scripts.synthesize import synthesize

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

CSS = """
/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    background: #080c18 !important;
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
    color: #e2e8f0 !important;
    min-height: 100vh;
}

/* ── Header bar ── */
.vc-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 40px;
    background: rgba(255,255,255,0.03);
    border-bottom: 1px solid rgba(255,255,255,0.07);
    backdrop-filter: blur(12px);
}
.vc-logo {
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #6C63FF, #00D4AA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.vc-badge {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #00D4AA;
    border: 1px solid rgba(0,212,170,0.35);
    padding: 4px 10px;
    border-radius: 20px;
    background: rgba(0,212,170,0.08);
}

/* ── Hero section ── */
.vc-hero {
    text-align: center;
    padding: 72px 24px 56px;
    background: radial-gradient(ellipse 80% 60% at 50% -10%, rgba(108,99,255,0.18) 0%, transparent 70%);
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
    color: #6C63FF;
    margin-bottom: 16px;
    display: block;
}
.vc-headline {
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.1;
    background: linear-gradient(135deg, #ffffff 0%, #a8b8d8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 20px;
}
.vc-headline span {
    background: linear-gradient(90deg, #6C63FF, #00D4AA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.vc-subline {
    font-size: 1.1rem;
    color: #8896aa;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Main content card ── */
.vc-main {
    max-width: 820px;
    margin: 0 auto;
    padding: 0 24px 80px;
}

/* ── Tab overrides ── */
.tab-nav {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    padding: 6px !important;
    gap: 4px !important;
    margin-bottom: 24px !important;
}
.tab-nav button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.01em !important;
    padding: 12px 28px !important;
    color: #8896aa !important;
    border: none !important;
    background: transparent !important;
    transition: all 0.2s ease !important;
}
.tab-nav button.selected,
.tab-nav button[aria-selected="true"] {
    background: linear-gradient(135deg, #6C63FF, #5a52e8) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 20px rgba(108,99,255,0.4) !important;
}

/* ── Input cards ── */
.vc-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 16px;
    transition: border-color 0.2s;
}
.vc-card:hover { border-color: rgba(108,99,255,0.3); }

/* ── Gradio input field overrides ── */
label > span, .block > label > span {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: #6872a0 !important;
    margin-bottom: 8px !important;
}
textarea, input[type="text"], select, .gr-input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #6C63FF !important;
    box-shadow: 0 0 0 3px rgba(108,99,255,0.15) !important;
}

/* ── Buttons ── */
.vc-btn-primary {
    background: linear-gradient(135deg, #6C63FF 0%, #5a52e8 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.02em !important;
    padding: 14px 36px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 24px rgba(108,99,255,0.35) !important;
    margin-top: 8px !important;
}
.vc-btn-primary:hover {
    background: linear-gradient(135deg, #7C73FF 0%, #6C63FF 100%) !important;
    box-shadow: 0 6px 32px rgba(108,99,255,0.55) !important;
    transform: translateY(-1px) !important;
}

/* ── Output section ── */
.vc-output-section {
    background: rgba(0,212,170,0.04);
    border: 1px solid rgba(0,212,170,0.15);
    border-radius: 16px;
    padding: 28px;
    margin-top: 8px;
}
.vc-status {
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    border: none !important;
    background: rgba(255,255,255,0.04) !important;
    color: #8896aa !important;
    padding: 10px 14px !important;
}

/* ── Audio component ── */
.waveform-container, audio {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* ── Divider ── */
.vc-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    margin: 32px 0;
}

/* ── Footer ── */
.vc-footer {
    text-align: center;
    padding: 32px 24px;
    color: #3d4a5c;
    font-size: 0.8rem;
    border-top: 1px solid rgba(255,255,255,0.05);
}
.vc-footer a { color: #6C63FF; text-decoration: none; }

/* ── Checkbox ── */
input[type="checkbox"]:checked {
    accent-color: #6C63FF;
}

/* ── File upload area ── */
.upload-container, .file-preview {
    background: rgba(255,255,255,0.03) !important;
    border: 2px dashed rgba(108,99,255,0.3) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s !important;
}
.upload-container:hover {
    border-color: rgba(108,99,255,0.6) !important;
    background: rgba(108,99,255,0.05) !important;
}

/* ── Microphone record button ── */
.vc-mic-card .mic-wrap,
.vc-mic-card [data-testid="waveform-record-button"],
.vc-mic-card button.record,
.vc-mic-card button[aria-label*="ecord"],
.vc-mic-card button[aria-label*="top"] {
    background: linear-gradient(135deg, #6C63FF, #5a52e8) !important;
    border: none !important;
    border-radius: 50% !important;
    width: 64px !important;
    height: 64px !important;
    min-width: 64px !important;
    color: #fff !important;
    cursor: pointer !important;
    box-shadow: 0 4px 20px rgba(108,99,255,0.5) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.vc-mic-card button[aria-label*="ecord"]:hover,
.vc-mic-card button[aria-label*="top"]:hover {
    transform: scale(1.08) !important;
    box-shadow: 0 6px 28px rgba(108,99,255,0.7) !important;
}
.vc-mic-card button[aria-label*="ecord"] svg,
.vc-mic-card button[aria-label*="top"] svg {
    width: 26px !important;
    height: 26px !important;
    fill: #fff !important;
    stroke: #fff !important;
}

/* ── gr.Group removes default border/bg — re-apply card styles ── */
.vc-card > div, .vc-mic-card > div, .vc-output-section > div {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
"""


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


LANG_CHOICES = [(name, code) for name, code in LANGUAGES]

HEADER_HTML = """
<div class="vc-header">
  <span class="vc-logo">VoiceClone AI</span>
  <span class="vc-badge">Powered by XTTS v2</span>
</div>
"""

HERO_HTML = """
<div class="vc-hero">
  <span class="vc-eyebrow">No account required &nbsp;·&nbsp; No credit card &nbsp;·&nbsp; No limits</span>
  <h1 class="vc-headline">FREE AI<br><span>VOICE CLONING</span></h1>
  <p class="vc-subline">
    Record your voice or upload any audio file, type your script,
    and hear it spoken back in your cloned voice — instantly.
  </p>
</div>
"""

with gr.Blocks(title="VoiceClone AI — Free AI Voice Cloning", css=CSS, theme=gr.themes.Base()) as demo:

    gr.HTML(HEADER_HTML)
    gr.HTML(HERO_HTML)

    with gr.Column(elem_classes="vc-main"):

        with gr.Tabs():

            # ── Tab 1: Microphone ──────────────────────────────────────
            with gr.Tab("  Clone Your Voice  "):
                with gr.Group(elem_classes="vc-card vc-mic-card"):
                    mic_audio = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="Record Your Voice  (6–30 seconds for best results)",
                        waveform_options={"waveform_color": "#6C63FF", "waveform_progress_color": "#00D4AA"},
                    )

                with gr.Group(elem_classes="vc-card"):
                    mic_text = gr.Textbox(
                        label="Text to Synthesize",
                        lines=5,
                        placeholder="Type the words you want spoken in your cloned voice...",
                    )
                    mic_lang = gr.Dropdown(
                        choices=LANG_CHOICES,
                        value="en",
                        label="Language",
                        interactive=True,
                    )

                mic_btn = gr.Button("Generate Speech", elem_classes="vc-btn-primary")

            # ── Tab 2: File Upload ─────────────────────────────────────
            with gr.Tab("  Clone from Audio File  "):
                with gr.Group(elem_classes="vc-card"):
                    file_upload = gr.File(
                        label="Upload MP4 or WAV File",
                        file_types=[".mp4", ".wav"],
                    )
                    if HAS_DEMUCS:
                        isolate_chk = gr.Checkbox(
                            label="Isolate vocals (remove background music/noise)",
                            value=False,
                        )
                    else:
                        isolate_chk = gr.Checkbox(
                            label="Isolate vocals (install demucs to enable)",
                            value=False,
                            interactive=False,
                        )

                with gr.Group(elem_classes="vc-card"):
                    file_text = gr.Textbox(
                        label="Text to Synthesize",
                        lines=5,
                        placeholder="Type the words you want spoken in the cloned voice...",
                    )
                    file_lang = gr.Dropdown(
                        choices=LANG_CHOICES,
                        value="en",
                        label="Language",
                        interactive=True,
                    )

                file_btn = gr.Button("Generate Speech", elem_classes="vc-btn-primary")

        # ── Output section ────────────────────────────────────────────
        gr.HTML('<div class="vc-divider"></div>')
        with gr.Group(elem_classes="vc-output-section"):
            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                elem_classes="vc-status",
                placeholder="Output will appear here after generation...",
            )
            audio_out = gr.Audio(label="Generated Audio", type="filepath")

    gr.HTML("""
    <div class="vc-footer">
      Built with <a href="https://github.com/coqui-ai/TTS" target="_blank">Coqui XTTS v2</a>
      &nbsp;·&nbsp; Voice samples stay on your machine — nothing is uploaded
    </div>
    """)

    mic_btn.click(
        fn=run_mic,
        inputs=[mic_audio, mic_text, mic_lang],
        outputs=[audio_out, status_box],
    )
    file_btn.click(
        fn=run_file,
        inputs=[file_upload, isolate_chk, file_text, file_lang],
        outputs=[audio_out, status_box],
    )

demo.launch()
