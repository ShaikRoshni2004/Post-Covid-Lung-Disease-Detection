import json
from pathlib import Path

import streamlit as st

from config.settings import CONFUSION_MATRIX_PATH, EVALUATION_METRICS_PATH, REPORTS_DIR, TRAINING_HISTORY_PLOT_PATH


def render_sidebar() -> None:
    """Render the common sidebar for all pages."""
    st.sidebar.title("Navigation")
    st.sidebar.caption("Model status")
    st.sidebar.markdown("**Model:** VGG16 Transfer Learning")
    st.sidebar.markdown("**Image Size:** 224 × 224")
    st.sidebar.markdown("**Classes:** Actual detected classes or N/A")
    st.sidebar.markdown("**Model Status:** Ready or Not Trained")
    st.sidebar.info(
        "This system is developed for academic and research purposes only. It is not a medical diagnostic tool and should not replace evaluation by a qualified healthcare professional."
    )


render_sidebar()

st.title("Model Performance")

if not EVALUATION_METRICS_PATH.exists():
    st.warning("Train and evaluate the model to generate performance results.")
else:
    with EVALUATION_METRICS_PATH.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    st.subheader("Evaluation Metrics")
    columns = st.columns(5)
    columns[0].metric("Accuracy", f"{metrics.get('test_accuracy', 0):.4f}")
    columns[1].metric("Precision", f"{metrics.get('precision', 0):.4f}")
    columns[2].metric("Recall", f"{metrics.get('recall', 0):.4f}")
    columns[3].metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
    columns[4].metric("Test Loss", f"{metrics.get('test_loss', 0):.4f}")

    if CONFUSION_MATRIX_PATH.exists():
        st.subheader("Confusion Matrix")
        st.image(str(CONFUSION_MATRIX_PATH), use_container_width=True)
    else:
        st.info("Confusion matrix image is not available yet.")

    if TRAINING_HISTORY_PLOT_PATH.exists():
        st.subheader("Training Accuracy/Loss Curves")
        st.image(str(TRAINING_HISTORY_PLOT_PATH), use_container_width=True)
    else:
        st.info("Training history plot is not available yet.")

    report_path = REPORTS_DIR / "classification_report.txt"
    if report_path.exists():
        st.subheader("Classification Report")
        st.code(report_path.read_text(encoding="utf-8"), language="text")
    else:
        st.info("Classification report is not available yet.")
