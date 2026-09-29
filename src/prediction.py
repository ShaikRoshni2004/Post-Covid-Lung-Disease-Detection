from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf

from config.settings import CLASS_NAMES_PATH, MODEL_PATH
from src.preprocessing import preprocess_image_for_vgg16


def load_model_if_exists(model_path: str | Path = MODEL_PATH) -> tf.keras.Model:
    """Load the trained model if it exists."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError("Model has not been trained yet.")
    return tf.keras.models.load_model(model_path, compile=False)


def load_class_names(class_names_path: str | Path = CLASS_NAMES_PATH) -> List[str]:
    """Load class names from the saved JSON mapping."""
    path = Path(class_names_path)
    if not path.exists():
        raise FileNotFoundError("Class names file not found.")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        if "class_names" in data:
            return list(data["class_names"])
        if "labels" in data:
            return list(data["labels"])

    if isinstance(data, list):
        return [str(item) for item in data]

    raise ValueError("Unsupported class names format.")


def predict_image(image_path: str | Path) -> Dict[str, Any]:
    """Predict the class and confidence for a single chest X-ray image."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model has not been trained yet.")
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError("Class names file not found.")

    model = load_model_if_exists(MODEL_PATH)
    class_names = load_class_names(CLASS_NAMES_PATH)
    processed_image = preprocess_image_for_vgg16(image_path)
    probabilities = model.predict(processed_image, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    probability_map = {
        class_names[index]: float(probabilities[index]) for index in range(len(class_names))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probability_map,
    }
