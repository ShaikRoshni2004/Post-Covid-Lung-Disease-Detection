from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from config.settings import CONFUSION_MATRIX_PATH, EVALUATION_METRICS_PATH
from src.utils import ensure_directory, write_json


def evaluate_model(
    model,
    test_dataset,
    class_names: List[str],
) -> Tuple[dict, np.ndarray, str]:
    """Evaluate a trained model on the test dataset."""

    # Compile the loaded model before calling model.evaluate()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    y_true = []
    y_pred = []

    # Generate predictions
    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)

        y_true.extend(
            np.argmax(labels.numpy(), axis=1)
        )

        y_pred.extend(
            np.argmax(predictions, axis=1)
        )

    # Calculate loss and accuracy
    loss, accuracy = model.evaluate(
        test_dataset,
        verbose=0,
    )

    # Calculate precision, recall and F1 score
    precision, recall, f1_score, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            average="weighted",
            zero_division=0,
        )
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=False,
    )

    metrics = {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "num_classes": len(class_names),
        "class_names": class_names,
    }

    return metrics, cm, report


def save_confusion_matrix(
    confusion_matrix_values: np.ndarray,
    class_names: List[str],
    output_path: str | Path,
) -> None:
    """Save a confusion matrix plot."""

    output_path = Path(output_path)
    ensure_directory(output_path.parent)

    figure, axis = plt.subplots(figsize=(8, 6))

    image = axis.imshow(
        confusion_matrix_values,
        cmap="Blues",
    )

    figure.colorbar(image, ax=axis)

    axis.set_xticks(range(len(class_names)))
    axis.set_yticks(range(len(class_names)))

    axis.set_xticklabels(
        class_names,
        rotation=45,
        ha="right",
    )

    axis.set_yticklabels(class_names)

    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Confusion Matrix")

    # Add numbers inside confusion matrix cells
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            axis.text(
                j,
                i,
                str(confusion_matrix_values[i, j]),
                ha="center",
                va="center",
            )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_classification_report(
    report_text: str,
    output_path: str | Path,
) -> None:
    """Save the full classification report as text."""

    output_path = Path(output_path)

    ensure_directory(output_path.parent)

    output_path.write_text(
        report_text,
        encoding="utf-8",
    )


def save_evaluation_outputs(
    metrics: dict,
    confusion_matrix_values: np.ndarray,
    class_names: List[str],
    report_text: str,
    metrics_path: str | Path = EVALUATION_METRICS_PATH,
    confusion_path: str | Path = CONFUSION_MATRIX_PATH,
) -> None:
    """Save all evaluation results."""

    write_json(
        metrics_path,
        metrics,
    )

    save_confusion_matrix(
        confusion_matrix_values,
        class_names,
        confusion_path,
    )

    report_path = str(metrics_path).replace(
        "evaluation_metrics.json",
        "classification_report.txt",
    )

    save_classification_report(
        report_text,
        report_path,
    )