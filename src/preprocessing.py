from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from config.settings import IMAGE_SIZE


def validate_image_path(image_path: str | Path) -> Path:
    """Validate and normalize an input image path."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(f"Unsupported image format: {path.suffix}")

    try:
        with Image.open(path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Corrupted or unreadable image: {path}") from exc

    return path


def read_image_dimensions(image_path: str | Path) -> Tuple[int, int]:
    """Return image width and height in pixels."""
    path = validate_image_path(image_path)
    with Image.open(path) as image:
        width, height = image.size
    return width, height


def load_rgb_image(image_path: str | Path) -> Image.Image:
    """Open image as RGB to standardize grayscale and color inputs."""
    path = validate_image_path(image_path)
    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        return rgb_image


def preprocess_image_for_vgg16(image_path: str | Path) -> np.ndarray:
    """Preprocess one image for VGG16 inference using the same logic as training."""
    rgb_image = load_rgb_image(image_path)
    resized_image = rgb_image.resize(IMAGE_SIZE)
    array = np.asarray(resized_image, dtype=np.float32)
    preprocessed = tf.keras.applications.vgg16.preprocess_input(array)
    return np.expand_dims(preprocessed, axis=0)
