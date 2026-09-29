import streamlit as st


def render_sidebar() -> None:
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

st.title("About Project")

st.subheader("Project Title")
st.write("Post Covid Lung Disease Detection")

st.subheader("Problem Statement")
st.write(
    "The project aims to classify lung conditions from chest X-ray images using deep learning and transfer learning in an academic research setting."
)

st.subheader("Objectives")
st.markdown(
    "- Build a CNN-based chest X-ray classifier\n"
    "- Use VGG16 transfer learning\n"
    "- Train, evaluate, and save a model\n"
    "- Provide a visual and interactive Streamlit interface"
)

st.subheader("Motivation")
st.write(
    "The growing impact of respiratory diseases after COVID-19 increases the need for accessible academic research tools and model experimentation."
)

st.subheader("Existing System")
st.write(
    "Many imaging studies rely on manual interpretation by clinicians. This project demonstrates an automated academic workflow that supports data-driven research."
)

st.subheader("Proposed System")
st.write(
    "A VGG16-based CNN model is trained on chest X-ray images and used to classify lung conditions, with predictions shared as model outputs only."
)

st.subheader("System Architecture")
st.markdown(
    "Chest X-Ray Dataset → Data Preprocessing → Data Augmentation → CNN / VGG16 Feature Extraction → Transfer Learning → Classification Layer → Disease Class Prediction → Confidence Analysis → Evaluation and Visualization"
)

st.subheader("Technologies Used")
st.write("Python, TensorFlow, Keras, Streamlit, OpenCV, scikit-learn, NumPy, Pandas, Matplotlib, Plotly, Pillow")

st.subheader("Deep Learning Method")
st.write(
    "A VGG16 convolutional neural network is used as the primary feature extractor, followed by a classification head trained for multi-class disease recognition."
)

st.subheader("VGG16 Explanation")
st.write(
    "VGG16 is a CNN architecture with repeated convolutional blocks and pooling layers. In this project, it serves as a pretrained feature extractor for chest X-ray classification."
)

st.subheader("Transfer Learning Explanation")
st.write(
    "Transfer learning reuses a model pretrained on ImageNet, which helps improve performance even when the target medical dataset is limited."
)

st.subheader("Data Augmentation Explanation")
st.write(
    "Augmentation adds realistic variations such as small rotations, translations, and zooms to improve learning robustness without distorting the X-ray too aggressively."
)

st.subheader("RetinaNet Clarification")
st.write(
    "RetinaNet is primarily an object-detection architecture and requires bounding-box annotations. This project is focused on image classification using VGG16 and a standard class-labeled dataset."
)

st.subheader("Workflow")
st.markdown(
    "- Collect and validate chest X-ray dataset\n"
    "- Preprocess images to 224 × 224\n"
    "- Train VGG16 transfer-learning model\n"
    "- Evaluate performance with metrics and confusion matrix\n"
    "- Run inference through the Streamlit app"
)

st.subheader("Advantages")
st.markdown(
    "- Reusable transfer-learning architecture\n"
    "- Transparent academic workflow\n"
    "- Local deployment in VS Code\n"
    "- Real dataset compatibility with automatic class detection"
)

st.subheader("Limitations")
st.write(
    "This is an academic prototype and should not be treated as a clinical diagnostic system. Model performance depends on the dataset and quality of labels."
)

st.subheader("Future Scope")
st.write(
    "Future work can include additional datasets, explainability tools, better curation, and more rigorous clinical validation."
)

st.subheader("Medical Disclaimer")
st.warning(
    "This system is developed for academic and research purposes only. It is not a medical diagnostic tool and should not replace evaluation by a qualified healthcare professional."
)
