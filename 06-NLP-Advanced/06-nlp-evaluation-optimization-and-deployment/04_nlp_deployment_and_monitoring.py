"""
NLP Deployment and Monitoring

Experiments:
- Build a lightweight NLP inference API with FastAPI.
- Validate incoming text and enforce input-size limits.
- Return structured classification predictions.
- Track request latency, errors, and prediction distributions.
- Expose health and monitoring endpoints.
- Test the service with a local HTTP client.
"""

from __future__ import annotations

import logging
import os
import time
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

MODEL_NAME = os.getenv(
    "NLP_MODEL_NAME",
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)
MAX_TEXT_LENGTH = 5_000
MAX_BATCH_SIZE = 16

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("nlp_service")


@dataclass
class MonitoringState:
    """Maintain process-local service metrics."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    prediction_counts: Counter[str] = field(default_factory=Counter)
    lock: Lock = field(default_factory=Lock)

    def record_success(
        self,
        latency_ms: float,
        labels: list[str],
    ) -> None:
        """Record a successful prediction request."""
        with self.lock:
            self.total_requests += 1
            self.successful_requests += 1
            self.total_latency_ms += latency_ms
            self.prediction_counts.update(labels)

    def record_failure(self) -> None:
        """Record a failed prediction request."""
        with self.lock:
            self.total_requests += 1
            self.failed_requests += 1

    def snapshot(self) -> dict[str, Any]:
        """Return a consistent snapshot of current metrics."""
        with self.lock:
            average_latency = (
                self.total_latency_ms / self.successful_requests
                if self.successful_requests
                else 0.0
            )

            return {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "average_latency_ms": round(average_latency, 2),
                "prediction_counts": dict(self.prediction_counts),
            }


class PredictionRequest(BaseModel):
    """Validate one or more texts submitted for classification."""

    texts: list[str] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class Prediction(BaseModel):
    """Represent one text classification result."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """Represent a prediction response."""

    predictions: list[Prediction]
    latency_ms: float
    model: str


class NLPService:
    """Load a classifier and expose prediction behavior."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.device = self._select_device()
        self.tokenizer: Any = None
        self.model: Any = None

    @staticmethod
    def _select_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        return torch.device("cpu")

    def load(self) -> None:
        """Load model artifacts into memory."""
        logger.info(
            "Loading model=%s device=%s",
            self.model_name,
            self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        self.model.to(self.device)
        self.model.eval()

        logger.info("Model loaded successfully")

    def predict(self, texts: list[str]) -> list[Prediction]:
        """Classify a batch of texts."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model has not been loaded")

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            logits = self.model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)

        scores, indices = probabilities.max(dim=-1)
        id_to_label = self.model.config.id2label

        predictions: list[Prediction] = []

        for score, index in zip(scores, indices):
            label = id_to_label.get(
                int(index),
                id_to_label.get(str(int(index)), str(int(index))),
            )
            predictions.append(
                Prediction(
                    label=str(label),
                    confidence=float(score),
                )
            )

        return predictions


service = NLPService(MODEL_NAME)
monitoring = MonitoringState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the model once when the service starts."""
    service.load()
    yield
    logger.info("NLP service shutting down")


app = FastAPI(
    title="NLP Classification Service",
    version="1.0.0",
    description="A small NLP inference API with basic monitoring.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Report process health and model initialization status."""
    loaded = service.model is not None

    if not loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready",
        )

    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": str(service.device),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    payload: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    """Validate inputs, predict labels, and record request metrics."""
    texts = payload.texts

    if any(not text.strip() for text in texts):
        monitoring.record_failure()
        raise HTTPException(
            status_code=422,
            detail="Texts must not be empty or whitespace-only",
        )

    if any(len(text) > MAX_TEXT_LENGTH for text in texts):
        monitoring.record_failure()
        raise HTTPException(
            status_code=413,
            detail=f"Each text must be at most {MAX_TEXT_LENGTH} characters",
        )

    started = time.perf_counter()

    try:
        predictions = service.predict(texts)
    except Exception:
        monitoring.record_failure()
        logger.exception(
            "Prediction failed method=%s path=%s",
            request.method,
            request.url.path,
        )
        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        ) from None

    latency_ms = (time.perf_counter() - started) * 1000
    labels = [prediction.label for prediction in predictions]

    monitoring.record_success(latency_ms, labels)

    logger.info(
        "Prediction completed batch_size=%d latency_ms=%.2f labels=%s",
        len(texts),
        latency_ms,
        labels,
    )

    return PredictionResponse(
        predictions=predictions,
        latency_ms=round(latency_ms, 2),
        model=MODEL_NAME,
    )


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    """Expose simple process-local monitoring metrics."""
    return monitoring.snapshot()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "04_nlp_deployment_and_monitoring:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
