from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import tensorflow as tf

from config.settings import (
    BATCH_SIZE,
    DATASET_DIR,
    IMAGE_SIZE,
    RANDOM_SEED,
    TEST_DIR,
    TRAIN_DIR,
    VALIDATION_DIR,
)
from src.utils import list_image_files


def discover_class_names(train_dir: str | Path = TRAIN_DIR) -> List[str]:
    """Discover class names from subdirectories in the training split."""
    directory = Path(train_dir)
    if not directory.exists():
        return []

    return sorted([path.name for path in directory.iterdir() if path.is_dir()])


def validate_dataset_structure(dataset_dir: str | Path = DATASET_DIR) -> dict:
    """Validate the dataset directories and image presence."""
    root = Path(dataset_dir)
    if not root.exists():
        return {
            "valid": False,
            "message": "Dataset directory not found. Please place the chest X-ray dataset in the dataset folder.",
            "class_names": [],
        }

    required_dirs = {
        "train": TRAIN_DIR,
        "validation": VALIDATION_DIR,
        "test": TEST_DIR,
    }

    for split_name, split_dir in required_dirs.items():
        if not split_dir.exists():
            return {
                "valid": False,
                "message": f"Missing {split_name} split directory: {split_dir}",
                "class_names": [],
            }

    class_names = discover_class_names(TRAIN_DIR)
    if len(class_names) < 2:
        return {
            "valid": False,
            "message": "At least two class folders are required in the training set.",
            "class_names": class_names,
        }

    for class_name in class_names:
        class_dir = TRAIN_DIR / class_name
        if not class_dir.exists():
            return {
                "valid": False,
                "message": f"Training class directory is missing: {class_dir}",
                "class_names": class_names,
            }
        image_files = list_image_files(class_dir)
        if not image_files:
            return {
                "valid": False,
                "message": f"Training class folder contains no valid images: {class_dir}",
                "class_names": class_names,
            }

    validation_names = discover_class_names(VALIDATION_DIR)
    test_names = discover_class_names(TEST_DIR)
    if set(class_names) != set(validation_names) or set(class_names) != set(test_names):
        return {
            "valid": False,
            "message": "Class folders in train, validation, and test splits do not match.",
            "class_names": class_names,
        }

    for split_name, split_dir in {"validation": VALIDATION_DIR, "test": TEST_DIR}.items():
        for class_name in class_names:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                return {
                    "valid": False,
                    "message": f"Missing {split_name} class folder: {class_dir}",
                    "class_names": class_names,
                }
            if not list_image_files(class_dir):
                return {
                    "valid": False,
                    "message": f"{split_name.title()} class folder contains no valid images: {class_dir}",
                    "class_names": class_names,
                }

    return {"valid": True, "message": "Dataset structure is valid.", "class_names": class_names}


def create_dataset_from_directory(
    directory: str | Path,
    class_names: List[str],
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    seed: int = RANDOM_SEED,
) -> tf.data.Dataset:
    """Create a tf.data dataset from a class-directory split."""
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        color_mode="rgb",
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    AUTOTUNE = tf.data.AUTOTUNE
    return dataset.prefetch(buffer_size=AUTOTUNE)


def load_dataset_splits(
    train_dir: str | Path = TRAIN_DIR,
    validation_dir: str | Path = VALIDATION_DIR,
    test_dir: str | Path = TEST_DIR,
    batch_size: int = BATCH_SIZE,
) -> Tuple[List[str], tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load train, validation, and test datasets for a valid dataset layout."""
    status = validate_dataset_structure(DATASET_DIR)
    if not status["valid"]:
        raise ValueError(status["message"])

    class_names = status["class_names"]
    train_ds = create_dataset_from_directory(train_dir, class_names, batch_size=batch_size, shuffle=True)
    val_ds = create_dataset_from_directory(validation_dir, class_names, batch_size=batch_size, shuffle=False)
    test_ds = create_dataset_from_directory(test_dir, class_names, batch_size=batch_size, shuffle=False)
    return class_names, train_ds, val_ds, test_ds
