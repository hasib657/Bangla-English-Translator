"""
FastAPI: English → Bangla Machine Translation API
==================================================

A REST API + premium web UI for English→Bangla translation using the
fine-tuned OPUS-MT model.

Endpoints:
    GET  /             — Premium web translation interface
    GET  /health       — Health check / API info
    POST /translate    — Translate English text to Bangla
    GET  /docs         — Interactive Swagger documentation

Deploy to Hugging Face Spaces:
    1. Create a new Space at https://huggingface.co/new-space
    2. Choose "Docker" as the SDK (or use the provided Dockerfile)
    3. Push this directory to the Space repository

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 7860
"""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "shhossain/opus-mt-en-to-bn")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 128))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Global model holders (populated at startup)
# ---------------------------------------------------------------------------
tokenizer: AutoTokenizer | None = None
model: AutoModelForSeq2SeqLM | None = None


# ---------------------------------------------------------------------------
# Lifespan — load model once and share across requests
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    global tokenizer, model

    print(f"Loading model '{MODEL_NAME}' on {DEVICE} …")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.1f}s — ready to serve requests.")

    yield  # app runs here

    # Shutdown cleanup
    print("Shutting down — releasing model resources.")
    del model, tokenizer


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="English → Bangla Translation API",
    description="Translate English text to Bangla using a fine-tuned OPUS-MT model.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow cross-origin requests (for browser-based clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (premium frontend)
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class TranslateRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="English text to translate.",
        examples=["The man is walking in the park."],
    )
    num_beams: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Beam-search width. Higher values may improve quality at the cost of speed.",
    )
    max_length: int = Field(
        default=128,
        ge=4,
        le=512,
        description="Maximum token length of the generated translation.",
    )


class TranslateResponse(BaseModel):
    source: str = Field(..., description="Original English text.")
    translation: str = Field(..., description="Bangla translation.")
    model: str = Field(..., description="Model used for translation.")
    inference_time_ms: float = Field(..., description="Inference latency in milliseconds.")


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_class=FileResponse)
async def home():
    """Serve the premium translation web interface."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(index_path)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check — returns API status and model info."""
    return HealthResponse(
        status="healthy",
        model=MODEL_NAME,
        device=str(DEVICE),
    )


@app.post("/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest):
    """
    Translate English text to Bangla.

    - **text**: The English sentence(s) to translate.
    - **num_beams**: Beam-search width (default 4).
    - **max_length**: Max output tokens (default 128).
    """
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Tokenize
    inputs = tokenizer(
        request.text.strip(),
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    # Generate
    t_start = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=request.max_length,
            num_beams=request.num_beams,
            early_stopping=True,
        )
    inference_time_ms = (time.time() - t_start) * 1000

    # Decode
    translation = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return TranslateResponse(
        source=request.text,
        translation=translation,
        model=MODEL_NAME,
        inference_time_ms=round(inference_time_ms, 2),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
