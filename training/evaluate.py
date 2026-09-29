from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf

from config.settings import (
    CLASS_NAMES_PATH,
    CONFUSION_MATRIX_PATH,
    DATASET_DIR,
    EVALUATION_METRICS_PATH,
    MODEL_PATH,
    PLOTS_DIR,
    REPORTS_DIR,
    TEST_DIR,
    TRAINING_HISTORY_PATH,
    TRAINING_HISTORY_PLOT_PATH,
    TRAINING_METADATA_PATH,
)
from src.dataset import validate_dataset_structure, load_dataset_splits
from src.evaluation import evaluate_model, save_confusion_matrix, save_evaluation_outputs
from src.utils import read_json, write_json


def main() -> int:
    """Load the trained model and evaluate it on the test-set split."""
    dataset_status = validate_dataset_structure(DATASET_DIR)
    if not dataset_status["valid"]:
        print(dataset_status["message"])
        return 0

    if not MODEL_PATH.exists():
        print("Trained model not found. Train the model before evaluation.")
        return 0

    if not CLASS_NAMES_PATH.exists():
        print("Class names file not found. Train the model before evaluation.")
        return 0

    try:
        class_names, _, _, test_ds = load_dataset_splits(DATASET_DIR / "train", DATASET_DIR / "validation", TEST_DIR)
    except ValueError as exc:
        print(str(exc))
        return 0

    test_ds = test_ds.map(
        lambda images, labels: (tf.keras.applications.vgg16.preprocess_input(images), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    metrics, confusion_matrix_values, report_text = evaluate_model(model, test_ds, class_names)

    write_json(EVALUATION_METRICS_PATH, metrics)
    save_confusion_matrix(confusion_matrix_values, class_names, CONFUSION_MATRIX_PATH)
    report_path = REPORTS_DIR / "classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    if TRAINING_HISTORY_PATH.exists():
        history_dict = read_json(TRAINING_HISTORY_PATH)
        if isinstance(history_dict, dict):
            try:
                from training.train import save_training_history_plot

                save_training_history_plot(history_dict, TRAINING_HISTORY_PLOT_PATH)
            except Exception:
                pass

    print(f"Test Loss: {metrics['test_loss']:.4f}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Evaluation metrics saved to: {EVALUATION_METRICS_PATH}")
    print(f"Classification report saved to: {report_path}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
