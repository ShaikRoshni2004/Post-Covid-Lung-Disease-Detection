import streamlit as st

from config.settings import CLASS_NAMES_PATH, EVALUATION_METRICS_PATH, MODEL_PATH
from src.prediction import load_class_names


def render_sidebar() -> None:
    """Render the common sidebar for all pages."""
    st.sidebar.title("Navigation")
    st.sidebar.caption("Model status")
    st.sidebar.markdown("**Model:** VGG16 Transfer Learning")
    st.sidebar.markdown("**Image Size:** 224 × 224")

    classes = "N/A"
    if CLASS_NAMES_PATH.exists():
        try:
            classes = ", ".join(load_class_names(CLASS_NAMES_PATH))
        except Exception:
            classes = "N/A"
    st.sidebar.markdown(f"**Classes:** {classes}")

    model_status = "Ready" if MODEL_PATH.exists() else "Not Trained"
    st.sidebar.markdown(f"**Model Status:** {model_status}")
    st.sidebar.info(
        "This system is developed for academic and research purposes only. It is not a medical diagnostic tool and should not replace evaluation by a qualified healthcare professional."
    )


render_sidebar()

st.title("Post Covid Lung Disease Detection")
st.caption("Deep Learning-Based Chest X-Ray Classification System")

st.markdown("### Model Architecture")
st.markdown("**VGG16 Transfer Learning**")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model Architecture", "VGG16")
with col2:
    st.metric("Input", "Chest X-Ray")
with col3:
    st.metric("Image Resolution", "224 × 224")

st.markdown("### PROJECT WORKFLOW")
st.markdown("Chest X-Ray → Preprocessing → VGG16 CNN → Feature Extraction → Classification → Prediction + Confidence")

if EVALUATION_METRICS_PATH.exists():
    import json

    with EVALUATION_METRICS_PATH.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    st.subheader("Evaluation results")
    metrics_columns = st.columns(5)
    metrics_columns[0].metric("Accuracy", f"{metrics.get('test_accuracy', 'N/A'):.4f}" if isinstance(metrics.get('test_accuracy'), (int, float)) else "N/A")
    metrics_columns[1].metric("Precision", f"{metrics.get('precision', 'N/A'):.4f}" if isinstance(metrics.get('precision'), (int, float)) else "N/A")
    metrics_columns[2].metric("Recall", f"{metrics.get('recall', 'N/A'):.4f}" if isinstance(metrics.get('recall'), (int, float)) else "N/A")
    metrics_columns[3].metric("F1 Score", f"{metrics.get('f1_score', 'N/A'):.4f}" if isinstance(metrics.get('f1_score'), (int, float)) else "N/A")
    metrics_columns[4].metric("Test Loss", f"{metrics.get('test_loss', 'N/A'):.4f}" if isinstance(metrics.get('test_loss'), (int, float)) else "N/A")
else:
    st.warning("Model evaluation results are not available yet.")
