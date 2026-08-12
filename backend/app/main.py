"""FastAPI server entry point."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.app.inference import TCMNetInference


DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:8081",
    "http://localhost:19006",
]


class PredictRequest(BaseModel):
    symptom_ids: list[str] = Field(default_factory=list)
    top_syndromes: int = Field(default=5, ge=1, le=25)
    top_herbs: int = Field(default=5, ge=1, le=25)


def get_allowed_origins() -> list[str]:
    raw_origins = os.getenv("BACKEND_CORS_ORIGINS")
    if not raw_origins:
        return DEFAULT_CORS_ORIGINS
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_inference() -> TCMNetInference:
    return TCMNetInference()


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_inference()
    yield


app = FastAPI(title="TCMNet Inference API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/symptoms")
def symptoms(inference: TCMNetInference = Depends(get_inference)) -> dict:
    return {
        "count": len(inference.symptoms_metadata),
        "symptoms": inference.symptoms_metadata,
    }


@app.get("/syndromes")
def syndromes(inference: TCMNetInference = Depends(get_inference)) -> dict:
    return {
        "count": len(inference.syndromes_metadata),
        "syndromes": inference.syndromes_metadata,
    }


@app.get("/herbs")
def herbs(inference: TCMNetInference = Depends(get_inference)) -> dict:
    return {
        "count": len(inference.herbs_metadata),
        "herbs": inference.herbs_metadata,
    }


@app.get("/concepts")
def concepts(inference: TCMNetInference = Depends(get_inference)) -> dict:
    return {
        "count": len(inference.concepts_metadata),
        "concepts": inference.concepts_metadata,
    }


@app.post("/predict")
def predict(
    request: PredictRequest,
    inference: TCMNetInference = Depends(get_inference),
) -> dict:
    return inference.predict(
        request.symptom_ids,
        top_syndromes=request.top_syndromes,
        top_herbs=request.top_herbs,
    )
