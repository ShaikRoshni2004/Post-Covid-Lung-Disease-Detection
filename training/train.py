from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from config.settings import (
    BATCH_SIZE,
    CLASS_NAMES_PATH,
    DATASET_DIR,
    EPOCHS,
    LEARNING_RATE,
    MODEL_DIR,
    MODEL_PATH,
    PLOTS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    RESULTS_DIR,
    TRAINING_HISTORY_PATH,
    TRAINING_HISTORY_PLOT_PATH,
    TRAINING_METADATA_PATH,
    TRAIN_DIR,
    VALIDATION_DIR,
)
from src.dataset import validate_dataset_structure, load_dataset_splits
from src.model_builder import build_vgg16_classifier, fine_tune_vgg16
from src.utils import ensure_directory, write_json


def save_training_history_plot(history_dict: Dict[str, list], output_path: str | Path) -> None:
    """Generate a training/validation accuracy and loss plot."""
    output_path = Path(output_path)
    ensure_directory(output_path.parent)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    if "accuracy" in history_dict:
        axes[0].plot(history_dict["accuracy"], label="Training Accuracy")
    if "val_accuracy" in history_dict:
        axes[0].plot(history_dict["val_accuracy"], label="Validation Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    if "loss" in history_dict:
        axes[1].plot(history_dict["loss"], label="Training Loss")
    if "val_loss" in history_dict:
        axes[1].plot(history_dict["val_loss"], label="Validation Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_training_metadata(class_names: list[str], epochs_requested: int, epochs_completed: int, model_type: str) -> None:
    """Save metadata about the training run."""
    metadata = {
        "image_size": [224, 224],
        "input_shape": [224, 224, 3],
        "number_of_classes": len(class_names),
        "class_names": class_names,
        "epochs_requested": epochs_requested,
        "epochs_completed": epochs_completed,
        "model_type": model_type,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "training_date": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(TRAINING_METADATA_PATH, metadata)


def main() -> int:
    """Validate dataset, train the VGG16 model, and save artifacts."""
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    ensure_directory(MODEL_DIR)
    ensure_directory(PLOTS_DIR)
    ensure_directory(REPORTS_DIR)

    print("Checking dataset structure...")
    dataset_status = validate_dataset_structure(DATASET_DIR)
    if not dataset_status["valid"]:
        print(dataset_status["message"])
        return 0

    class_names = dataset_status["class_names"]
    print(f"Detected classes: {class_names}")

    class_names_path = MODEL_DIR / "class_names.json"
    with class_names_path.open("w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)

    print("Loading dataset splits...")
    class_names_loaded, train_ds, val_ds, _ = load_dataset_splits(TRAIN_DIR, VALIDATION_DIR, DATASET_DIR / "test", BATCH_SIZE)
    train_ds = train_ds.map(
        lambda images, labels: (tf.keras.applications.vgg16.preprocess_input(images), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    val_ds = val_ds.map(
        lambda images, labels: (tf.keras.applications.vgg16.preprocess_input(images), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    model = build_vgg16_classifier(num_classes=len(class_names_loaded), learning_rate=LEARNING_RATE, freeze_base=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
    ]

    print("Training VGG16 transfer-learning model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    history_dict = history.history
    save_training_history_plot(history_dict, TRAINING_HISTORY_PLOT_PATH)
    write_json(TRAINING_HISTORY_PATH, history_dict)
    save_training_metadata(class_names_loaded, EPOCHS, len(history_dict.get("loss", [])), "VGG16 Transfer Learning")

    final_model_path = MODEL_DIR / "final_model.keras"
    model.save(final_model_path)
    print(f"Best model saved to: {MODEL_PATH}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Training history saved to: {TRAINING_HISTORY_PATH}")
    print(f"Metadata saved to: {TRAINING_METADATA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
