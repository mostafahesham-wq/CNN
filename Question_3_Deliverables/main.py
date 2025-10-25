import os
import io
import logging
from typing import List, Dict

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
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

# Path to frontend
FRONTEND_PATH = os.path.join(os.path.dirname(__file__), "..", "Question_4_Deliverables")

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


# Serve frontend files
@app.get("/")
async def serve_index():
    """Serve the main HTML page"""
    index_file = os.path.join(FRONTEND_PATH, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"error": "Frontend not found"}, status_code=404)


@app.get("/style.css")
async def serve_css():
    """Serve CSS file"""
    css_file = os.path.join(FRONTEND_PATH, "style.css")
    if os.path.exists(css_file):
        return FileResponse(css_file)
    return JSONResponse({"error": "CSS not found"}, status_code=404)


@app.get("/script.js")
async def serve_js():
    """Serve JavaScript file"""
    js_file = os.path.join(FRONTEND_PATH, "script.js")
    if os.path.exists(js_file):
        return FileResponse(js_file)
    return JSONResponse({"error": "JavaScript not found"}, status_code=404)


# Health check endpoint
@app.get("/health")
async def health():
    model = get_model()
    return {"status": "healthy", "model_loaded": model is not None}


# Prediction endpoint - MODIFIED to work with frontend (no auth required for demo)
@app.post("/predict")
async def predict_public(
    file: UploadFile = File(...),
):
    """
    Public prediction endpoint for frontend demo
    No authentication required - use /predict/secure for authenticated endpoint
    """
    model = get_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # validate file content-type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG or PNG)")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        logger.exception("Failed to open image: %s", e)
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        x = preprocess_pil(img, size=IMG_SIZE)
        preds = model.predict(x)
        probs = preds[0].tolist()

        # Build predictions list sorted by probability (for frontend)
        predictions_list = []
        for i in range(len(CLASS_NAMES)):
            predictions_list.append({
                "class": CLASS_NAMES[i],
                "prob": float(probs[i])
            })
        
        # Sort by probability descending
        predictions_list.sort(key=lambda x: x["prob"], reverse=True)

        # Get top prediction
        top_idx = int(np.argmax(probs))
        
        # Build predictions mapping (for compatibility)
        predictions_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

        result = {
            "class": CLASS_NAMES[top_idx],
            "confidence": float(probs[top_idx]),
            "predictions": predictions_list,  # Frontend expects this format
            "all_predictions": predictions_dict,  # Keep for compatibility
        }

        # log into DB (fire-and-forget)
        try:
            log_prediction(
                filename=file.filename or "unknown.png",
                predicted_class=CLASS_NAMES[top_idx],
                confidence=float(probs[top_idx]),
                raw_probs=predictions_dict
            )
        except Exception as e:
            logger.exception("Failed to log prediction to DB: %s", e)

        logger.info("Prediction made for %s -> %s (%.3f)", 
                   file.filename or "unknown", CLASS_NAMES[top_idx], float(probs[top_idx]))
        
        return JSONResponse(result)
    
    except Exception as e:
        logger.exception("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Secure prediction endpoint with authentication
@app.post("/predict/secure")
async def predict_secure(
    request: Request,
    file: UploadFile = File(...),
    api_key: str = Depends(get_api_key_header),
):
    """
    Secure prediction endpoint requiring API key
    Use X-API-KEY header with value 'dev-key-123' for testing
    """
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
        log_prediction(
            filename=file.filename,
            predicted_class=CLASS_NAMES[top_idx],
            confidence=float(probs[top_idx]),
            raw_probs=predictions
        )
    except Exception:
        logger.exception("Failed to log prediction to DB")

    logger.info("Secure prediction made for %s -> %s (%.3f)", 
               file.filename, CLASS_NAMES[top_idx], float(probs[top_idx]))
    return JSONResponse(result)


# Batch prediction endpoint (bonus)
@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    api_key: str = Depends(get_api_key_header),
):
    """
    Batch prediction endpoint requiring API key
    """
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
        result = {
            "filename": file.filename,
            "class": CLASS_NAMES[top_idx],
            "confidence": float(probs[top_idx]),
            "predictions": predictions
        }
        # log
        try:
            log_prediction(
                filename=file.filename,
                predicted_class=CLASS_NAMES[top_idx],
                confidence=float(probs[top_idx]),
                raw_probs=predictions
            )
        except Exception:
            logger.exception("Failed to log prediction")
        results.append(result)

    return JSONResponse({"results": results})


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)