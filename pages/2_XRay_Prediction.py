from pathlib import Path
import tempfile

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from config.settings import CLASS_NAMES_PATH, MODEL_PATH
from src.prediction import predict_image


def render_sidebar() -> None:
    """Render the common sidebar for all pages."""
    st.sidebar.title("Navigation")
    st.sidebar.caption("Model status")
    st.sidebar.markdown("**Model:** VGG16 Transfer Learning")
    st.sidebar.markdown("**Image Size:** 224 × 224")

    classes = "N/A"
    if CLASS_NAMES_PATH.exists():
        try:
            import json

            with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
                data = json.load(file)
            if isinstance(data, list):
                classes = ", ".join(data)
        except Exception:
            classes = "N/A"
    st.sidebar.markdown(f"**Classes:** {classes}")

    model_status = "Ready" if MODEL_PATH.exists() else "Not Trained"
    st.sidebar.markdown(f"**Model Status:** {model_status}")
    st.sidebar.info(
        "This system is developed for academic and research purposes only. It is not a medical diagnostic tool and should not replace evaluation by a qualified healthcare professional."
    )


render_sidebar()

st.title("Chest X-Ray Analysis")

uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    width, height = image.size
    st.image(image, caption="Uploaded chest X-ray", use_container_width=True)
    st.write(f"Image dimensions: {width} × {height}")

    if st.button("Analyze X-Ray"):
        if not MODEL_PATH.exists():
            st.error("Trained model not found. Train the model before running predictions.")
        else:
            try:
                suffix = Path(uploaded_file.name).suffix.lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = Path(temp_file.name)

                result = predict_image(temp_path)
                temp_path.unlink(missing_ok=True)

                st.subheader("PREDICTION RESULTS")
                st.write(f"Model-predicted class: {result['predicted_class']}")
                st.write(f"Confidence: {result['confidence'] * 100:.2f}%")

                st.subheader("CLASS PROBABILITIES")
                probability_df = pd.DataFrame(
                    {
                        "Class": list(result["probabilities"].keys()),
                        "Probability": [float(value) for value in result["probabilities"].values()],
                    }
                )
                probability_df = probability_df.sort_values("Probability", ascending=False)

                fig = px.bar(
                    probability_df,
                    x="Probability",
                    y="Class",
                    orientation="h",
                    color="Probability",
                    color_continuous_scale="Blues",
                    title="Class Probabilities",
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(probability_df, use_container_width=True)
            except Exception as exc:
                st.error(f"Unable to process the uploaded image: {exc}")

            st.warning(
                "This system is developed for academic and research purposes only. It is not a medical diagnostic tool and should not replace evaluation by a qualified healthcare professional."
            )
