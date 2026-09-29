import streamlit as st

st.set_page_config(
    page_title="Post Covid Lung Disease Detection",
    page_icon="🫁",
    layout="wide",
)

st.title("Post Covid Lung Disease Detection")
st.caption("Deep Learning-Based Chest X-Ray Classification System")

st.info(
    "This system is developed for academic and research purposes only. "
    "It is not a medical diagnostic tool and should not replace evaluation by a qualified healthcare professional."
)

st.markdown(
    """
    ### Overview
    This project demonstrates a VGG16 transfer-learning pipeline for chest X-ray classification.
    The software is designed to identify the most likely disease class from a chest X-ray image,
    while clearly labeling outputs as model predictions for academic use only.
    """
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model Architecture", "VGG16")
with col2:
    st.metric("Input", "Chest X-Ray")
with col3:
    st.metric("Image Size", "224 × 224")

st.subheader("Project Workflow")
st.markdown(
    "Chest X-Ray → Preprocessing → VGG16 CNN → Feature Extraction → Classification → Prediction + Confidence"
)

st.subheader("Status")
if st.button("Open the prediction page", use_container_width=True):
    st.switch_page("pages/2_XRay_Prediction.py")
