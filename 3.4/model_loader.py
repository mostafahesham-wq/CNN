import os
import logging
from typing import Optional

import tensorflow as tf
from tensorflow.keras.models import load_model

logger = logging.getLogger("model_loader")

_MODEL = None


def load_cifar_model(path: str):
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    # Try loading HDF5 or SavedModel dir
    try:
        _MODEL = load_model(path)
    except Exception as e:
        # fallback: try loading as SavedModel directory
        _MODEL = tf.keras.models.load_model(path)
    logger.info("Model loaded from %s", path)
    return _MODEL


def get_model():
    return _MODEL
