from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image as kimage


def preprocess_pil(img: Image.Image, size=(32, 32)) -> np.ndarray:
    """Convert PIL image -> model-ready numpy batch (1, H, W, C)

    - Resize with bilinear
    - Convert to RGB
    - Normalize to [0,1]
    """
    img = img.convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    arr = kimage.img_to_array(img)  # HWC
    arr = np.expand_dims(arr, axis=0)  # NHWC
    arr = arr / 255.0
    return arr
