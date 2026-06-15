# English → Bangla Translation API

REST API for English→Bangla machine translation using a fine-tuned OPUS-MT model.

## 🚀 Deploy to Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose **Docker** as the SDK (auto-detected from the Dockerfile) or use the API template
3. Push this `api/` directory content to the Space repo
4. The Space will build and deploy automatically

## 🔧 Run Locally

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

The API is available at http://localhost:7860.
Interactive docs at http://localhost:7860/docs.

## 📡 API Endpoints

### `GET /` — Health Check

```bash
curl http://localhost:7860/
```

Response:
```json
{
  "status": "healthy",
  "model": "shhossain/opus-mt-en-to-bn",
  "device": "cuda"
}
```

### `POST /translate` — Translate Text

```bash
curl -X POST http://localhost:7860/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "The man is walking in the park.", "num_beams": 4, "max_length": 128}'
```

Response:
```json
{
  "source": "The man is walking in the park.",
  "translation": "লোকটি পার্কে হাঁটছে",
  "model": "shhossain/opus-mt-en-to-bn",
  "inference_time_ms": 245.32
}
```

### Python Client Example

```python
import requests

resp = requests.post(
    "http://localhost:7860/translate",
    json={"text": "Hello, how are you?", "num_beams": 4},
)
print(resp.json()["translation"])
```

## 📦 Model

Default model: `shhossain/opus-mt-en-to-bn`.

Override via environment variable:
```bash
MODEL_NAME=your-username/your-fine-tuned-model uvicorn main:app --port 7860
```
