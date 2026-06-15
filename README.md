# Project 2: Machine Translation — English → Bangla

Fine-tuned OPUS-MT transformer model for English→Bangla machine translation with MLFlow experiment tracking, Gradio web app, and REST API deployment.

## 📁 Project Structure

```
.
├── project_2_mt_en_bn.ipynb   # Jupyter notebook with full pipeline + MLFlow
├── train.py                    # Standalone training script with MLFlow
├── requirements.txt            # All project dependencies
├── train.csv / val.csv / test.csv  # English→Bangla parallel corpus
├── gradio_app/                 # Gradio web interface (HF Spaces ready)
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
└── api/                        # FastAPI REST API (HF Spaces ready)
    ├── main.py
    ├── Dockerfile
    ├── requirements.txt
    └── README.md
```

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train with MLFlow Tracking

```bash
# Start MLFlow UI (in a separate terminal)
mlflow ui --port 5000

# Run training
python train.py --train_csv train.csv --val_csv val.csv --test_csv test.csv \
    --batch_size 16 --max_epochs 5 --experiment_name mt_en_bn_finetune
```

Training metrics (loss, BLEU score), hyperparameters, and model checkpoints are automatically logged to MLFlow.
Open http://localhost:5000 to view the experiment dashboard.

## 🚀 Hugging Face Spaces Deployment

### Gradio App

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Gradio** as the SDK
3. Push the `gradio_app/` directory content to the Space repository
4. The Space auto-builds and deploys

**Local test:**
```bash
cd gradio_app
pip install -r requirements.txt
python app.py
# → http://localhost:7860
```

### REST API

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Docker** as the SDK
3. Push the `api/` directory content to the Space repository

**Local test:**
```bash
cd api
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
# → API: http://localhost:7860
# → Docs: http://localhost:7860/docs
```

## 🤖 Model

Base model: [shhossain/opus-mt-en-to-bn](https://huggingface.co/shhossain/opus-mt-en-to-bn) (Helsinki-NLP OPUS-MT fine-tuned for English→Bangla).

After fine-tuning, upload your model to Hugging Face Hub and update the `MODEL_NAME` env variable in the Gradio app and API to point to your model.

## 📊 MLFlow Tracking

| What's Tracked | How |
|---|---|
| Hyperparameters | `batch_size`, `learning_rate`, `max_epochs`, etc. |
| Training metrics | `train_loss`, `val_loss`, `test_loss` |
| BLEU scores | `val_bleu`, `test_bleu` |
| Model checkpoints | Best 2 checkpoints (by `val_bleu`) |
| Final model | Full model + tokenizer as artifact |
| Learning rate | Per-epoch LR logging |
