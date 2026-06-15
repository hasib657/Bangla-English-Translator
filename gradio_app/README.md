# English → Bangla Machine Translation (Gradio App)

A Gradio web interface for translating English text to Bangla using a fine-tuned OPUS-MT transformer model.

## 🚀 Deploy to Hugging Face Spaces

[![Deploy to HF Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/deploy-to-spaces-github.svg)](https://huggingface.co/new-space)

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose **Gradio** as the SDK
3. Clone the Space repo and push this `gradio_app/` directory content
4. The Space will build and deploy automatically

## 🔧 Run Locally

```bash
cd gradio_app
pip install -r requirements.txt
python app.py
```

The app opens at http://localhost:7860.

## 📦 Model

Default model: `shhossain/opus-mt-en-to-bn` (OPUS-MT fine-tuned for English→Bangla).

To use your own fine-tuned model, set the `MODEL_NAME` environment variable:

```bash
MODEL_NAME=your-username/your-fine-tuned-model python app.py
```

## 📝 Example

| English | Bangla |
|---------|--------|
| The man is walking in the park. | লোকটি পার্কে হাঁটছে |
| A dog is playing with a ball. | একটি কুকুর বল নিয়ে খেলছে |
