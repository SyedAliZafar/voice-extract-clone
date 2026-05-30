import gradio as gr
from pathlib import Path
from scripts.synthesize import synthesize

VOICE_REF = Path("audio/extracted.wav")
OUTPUT = Path("output/result.wav")


def run(text_input, file_upload):
    text = text_input.strip() if text_input else ""
    if not text and file_upload:
        text = Path(file_upload).read_text(encoding="utf-8").strip()
    if not text:
        return None, "No text provided."
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    synthesize(VOICE_REF, text, OUTPUT)
    return str(OUTPUT), "Done."


with gr.Blocks(title="Voice Clone") as demo:
    gr.Markdown("## Voice Clone\nType your script or upload a `.txt` file. Voice is fixed.")
    with gr.Row():
        text_box = gr.Textbox(
            label="Type script here",
            lines=8,
            placeholder="Paste or type your text...",
        )
        file_box = gr.File(label="Or upload a .txt file", file_types=[".txt"])
    run_btn = gr.Button("Generate", variant="primary")
    status = gr.Textbox(label="Status", interactive=False)
    audio_out = gr.Audio(label="Output", type="filepath")
    run_btn.click(fn=run, inputs=[text_box, file_box], outputs=[audio_out, status])

demo.launch()
