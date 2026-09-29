from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.preprocessing import preprocess_image_for_vgg16, validate_image_path


def test_valid_image_preprocessing(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (224, 224), color=(255, 255, 255)).save(image_path)

    processed = preprocess_image_for_vgg16(image_path)

    assert isinstance(processed, np.ndarray)
    assert processed.shape == (1, 224, 224, 3)
    assert processed.dtype == np.float32


def test_invalid_image_handling(tmp_path):
    invalid_image = tmp_path / "broken.png"
    invalid_image.write_bytes(b"not-a-real-image")

    with pytest.raises(ValueError):
        validate_image_path(invalid_image)


def test_expected_image_output_dimensions(tmp_path):
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (512, 256), color=(50, 50, 50)).save(image_path)

    processed = preprocess_image_for_vgg16(image_path)

    assert processed.shape == (1, 224, 224, 3)
