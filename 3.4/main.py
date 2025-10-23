import os
import io
import logging
from typing import List, Dict

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_401_UNAUTHORIZED
import uvicorn
from PIL import Image
import numpy as np

from model_loader import load_cifar_model, get_model
from preprocessing import preprocess_pil
from auth import verify_api_key, get_api_key_header
from rate_limiter import rate_limit
from db import init_db, log_prediction

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cifar_api")

app = FastAPI(title="CIFAR-10 Classifier API")

# CORS (allow frontend to access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CONFIG
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.getcwd(), "best_Standard_CNN.h5"))
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
IMG_SIZE = (32, 32)

# Load model on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up: initializing DB and loading model...")
    init_db()
    try:
        load_cifar_model(MODEL_PATH)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.exception("Failed to load model at startup: %s", e)


# Health check endpoint
@app.get("/health")
async def health():
    model = get_model()
    return {"status": "healthy", "model_loaded": model is not None}


# Single prediction endpoint
@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    api_key: str = Depends(get_api_key_header),
):
    # rate limit (per API key)
    await rate_limit(api_key)

    model = get_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # validate file content-type
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="File must be JPEG or PNG")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    x = preprocess_pil(img, size=IMG_SIZE)
    preds = model.predict(x)
    probs = preds[0].tolist()

    # build predictions mapping
    predictions = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    top_idx = int(np.argmax(probs))
    result = {
        "class": CLASS_NAMES[top_idx],
        "confidence": float(probs[top_idx]),
        "predictions": predictions,
    }

    # log into DB (fire-and-forget)
    try:
        log_prediction(filename=file.filename, predicted_class=CLASS_NAMES[top_idx], confidence=float(probs[top_idx]), raw_probs=predictions)
    except Exception:
        logger.exception("Failed to log prediction to DB")

    logger.info("Prediction made for %s -> %s (%.3f)", file.filename, CLASS_NAMES[top_idx], float(probs[top_idx]))
    return JSONResponse(result)


# Batch prediction endpoint (bonus)
@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    api_key: str = Depends(get_api_key_header),
):
    await rate_limit(api_key)
    model = get_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    results = []
    for file in files:
        if file.content_type not in ("image/jpeg", "image/png"):
            results.append({"filename": file.filename, "error": "unsupported file type"})
            continue
        contents = await file.read()
        try:
            img = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            results.append({"filename": file.filename, "error": "invalid image"})
            continue
        x = preprocess_pil(img, size=IMG_SIZE)
        preds = model.predict(x)
        probs = preds[0].tolist()
        predictions = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        top_idx = int(np.argmax(probs))
        result = {"filename": file.filename, "class": CLASS_NAMES[top_idx], "confidence": float(probs[top_idx]), "predictions": predictions}
        # log
        try:
            log_prediction(filename=file.filename, predicted_class=CLASS_NAMES[top_idx], confidence=float(probs[top_idx]), raw_probs=predictions)
        except Exception:
            logger.exception("Failed to log prediction")
        results.append(result)

    return JSONResponse({"results": results})


if __name__ == "__main__":
    uvicorn.run("main:app", host="127")

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "CIFAR-10 FastAPI is running!"}
