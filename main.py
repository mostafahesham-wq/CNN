from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image
import uvicorn

# 1️⃣ Initialize FastAPI app
app = FastAPI(title="CIFAR-10 Image Classifier API")

# 2️⃣ Load trained model
MODEL_PATH = "cnn_model.h5"
model = load_model(MODEL_PATH)

# 3️⃣ CIFAR-10 class labels
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# 4️⃣ Define image size (CIFAR-10 = 32x32)
IMG_SIZE = (32, 32)

# 5️⃣ Prediction endpoint
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize(IMG_SIZE)

        # Preprocess image
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # normalization (CIFAR-10 models need this)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return JSONResponse({
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": confidence
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# 6️⃣ Run app locally
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
