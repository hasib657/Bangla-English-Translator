"""
Gradio App: English → Bangla Machine Translation
=================================================

A Gradio web interface for translating English text to Bangla using
the fine-tuned OPUS-MT model.

Deploy to Hugging Face Spaces:
    1. Create a new Space at https://huggingface.co/new-space
    2. Choose "Gradio" as the SDK
    3. Push this directory to the Space repository

Alternatively, run locally:
    python app.py
"""

import os
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------------------------------------------------------
# Configuration — change MODEL_NAME to your fine-tuned HF Hub model after training
# ---------------------------------------------------------------------------
# Use the pretrained model as default; replace with your fine-tuned model on HF Hub.
MODEL_NAME = os.environ.get("MODEL_NAME", "shhossain/opus-mt-en-to-bn")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 128))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Load model & tokenizer once at startup
# ---------------------------------------------------------------------------
print(f"Loading model '{MODEL_NAME}' on {DEVICE} …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print("Model loaded successfully.")

# ---------------------------------------------------------------------------
# Translation function
# ---------------------------------------------------------------------------
def translate(text: str, num_beams: int = 4, max_length: int = MAX_LENGTH) -> str:
    """
    Translate the given English text to Bangla.

    Args:
        text: English input sentence(s).
        num_beams: Beam-search width (higher = potentially better, but slower).
        max_length: Maximum length of the generated translation.
    """
    if not text or not text.strip():
        return "⚠️  Please enter some English text to translate."

    # Tokenize
    inputs = tokenizer(
        text.strip(),
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )

    # Decode
    translation = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return translation


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------
def build_interface():
    """Build and return the Gradio Blocks interface."""

    # Example English sentences for the user to try
    examples = [
        ["The man is walking in the park."],
        ["A dog is playing with a ball."],
        ["She is reading a book under the tree."],
        ["The children are dancing in the rain."],
        ["I love eating traditional Bengali food."],
    ]

    with gr.Blocks(title="English → Bangla Translator") as demo:
        gr.Markdown(
            """
            # 🇧🇩 English → Bangla Machine Translation

            Translate English sentences into **Bangla (Bengali)** using a
            fine-tuned **OPUS-MT** transformer model.

            Enter your English text below and click **Translate**.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_text = gr.Textbox(
                    label="📝 English Input",
                    placeholder="Type your English sentence here …",
                    lines=4,
                    max_lines=8,
                )

                with gr.Row():
                    beams = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=4,
                        step=1,
                        label="🔎 Beam Search Width",
                    )
                    max_len = gr.Slider(
                        minimum=16,
                        maximum=256,
                        value=128,
                        step=8,
                        label="📏 Max Output Length",
                    )

                translate_btn = gr.Button("✨ Translate", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_text = gr.Textbox(
                    label="🇧🇩 Bangla Translation",
                    placeholder="Translation will appear here …",
                    lines=4,
                    max_lines=8,
                    elem_classes=["output-box"],
                )

        # Click handler
        translate_btn.click(
            fn=translate,
            inputs=[input_text, beams, max_len],
            outputs=output_text,
        )

        # Also translate on Enter (via submit)
        input_text.submit(
            fn=translate,
            inputs=[input_text, beams, max_len],
            outputs=output_text,
        )

        # Examples
        gr.Markdown("### Try these examples:")
        gr.Examples(
            examples=examples,
            inputs=input_text,
            outputs=output_text,
            fn=translate,
            cache_examples=False,
        )

        # Footer
        gr.Markdown(
            """
            ---
            🔧 Built with [Gradio](https://gradio.app) ·
            🤗 Deployed on [Hugging Face Spaces](https://huggingface.co/spaces) ·
            📦 Model: `shhossain/opus-mt-en-to-bn`
            """
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo = build_interface()
    print("Starting Gradio server...")
    demo.queue(max_size=16).launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css="""
            .output-box textarea { font-size: 1.3rem !important; }
            footer { visibility: hidden; }
        """,
    )
