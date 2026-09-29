# Post Covid Lung Disease Detection

A deep learning project for chest X-ray image classification using VGG16 transfer learning. The goal is to build a practical academic prototype that can detect disease classes from chest X-ray images while clearly stating that it is not a medical diagnostic tool.

## Overview

This project demonstrates how to build a convolutional neural network workflow for image classification using TensorFlow/Keras. It uses a VGG16 backbone pretrained on ImageNet and adapts it to chest X-ray classification through transfer learning. The application is implemented as a Streamlit web app and is designed for local use in VS Code on Windows.

## Problem Statement

The post-COVID era has increased interest in respiratory disease detection from chest X-rays. Automated image classification can help researchers analyze large medical imaging datasets and compare model behavior. This project focuses on the academic and research use case, not direct clinical diagnosis.

## Objectives

- Detect disease classes from chest X-ray images
- Implement VGG16 transfer learning
- Train and evaluate a CNN-based classifier
- Provide image preprocessing and augmentation pipeline
- Save model and metadata for reuse
- Deliver a Streamlit interface for local use
- Present outputs with a clear academic medical disclaimer

## Features

- Multi-class chest X-ray image classification
- Automatic class discovery from dataset folder names
- VGG16 transfer learning with configurable fine-tuning
- Light augmentation during training
- Model checkpointing and early stopping
- Evaluation metrics, confusion matrix, and classification report
- Streamlit dashboard, prediction page, performance page, and about page
- No fabricated clinical claims or fake results

## System Architecture

Chest X-Ray Dataset
↓
Data Preprocessing
↓
Data Augmentation
↓
CNN / VGG16 Feature Extraction
↓
Transfer Learning
↓
Classification Layer
↓
Disease Class Prediction
↓
Confidence Analysis
↓
Evaluation and Visualization

## Project Workflow

1. Prepare chest X-ray dataset
2. Validate directory structure and class folders
3. Load and preprocess images
4. Train VGG16 transfer model
5. Save model and class metadata
6. Evaluate on the test set
7. Run inference through Streamlit

## Technology Stack

- Python 3.11 / 3.12
- TensorFlow/Keras
- VGG16
- Streamlit
- OpenCV
- scikit-learn
- Matplotlib
- Plotly
- PIL
- NumPy
- Pandas
- Pytest

## Project Structure

- [app.py](app.py)
- [requirements.txt](requirements.txt)
- [README.md](README.md)
- [config/settings.py](config/settings.py)
- [src/preprocessing.py](src/preprocessing.py)
- [src/dataset.py](src/dataset.py)
- [src/model_builder.py](src/model_builder.py)
- [src/prediction.py](src/prediction.py)
- [training/train.py](training/train.py)
- [training/evaluate.py](training/evaluate.py)
- [pages/1_Dashboard.py](pages/1_Dashboard.py)
- [pages/2_XRay_Prediction.py](pages/2_XRay_Prediction.py)
- [pages/3_Model_Performance.py](pages/3_Model_Performance.py)
- [pages/4_About_Project.py](pages/4_About_Project.py)

## Dataset Setup

This project expects a chest X-ray dataset organized with train, validation, and test folders, for example:

```text
dataset/
    train/
        class_1/
        class_2/
    validation/
        class_1/
        class_2/
    test/
        class_1/
        class_2/
```

The code discovers class names automatically from the training directories. The dataset is intentionally not bundled in GitHub because medical imaging datasets can be large and may have licensing restrictions.

## Installation

### Windows PowerShell

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Training

Run training from the project root:

```powershell
python training/train.py
```

This script validates the dataset, discovers labels, builds the VGG16 model, trains it with callbacks, saves the best model, and stores evaluation metadata.

## Evaluation

Run evaluation from the project root:

```powershell
python training/evaluate.py
```

This script loads the trained model, evaluates it on the test set, and saves metrics, a confusion matrix, and classification results.

## Running the Application

```powershell
streamlit run app.py
```

## Prediction Workflow

1. Upload a chest X-ray image in the Streamlit app
2. View the uploaded image and dimensions
3. Run prediction
4. Read the predicted class and confidence score
5. Review the class probability bar chart and probability table

## Model Architecture

This project uses VGG16 as the primary CNN backbone.

- VGG16 is itself a convolutional neural network architecture
- The pretrained ImageNet weights are used for transfer learning
- The original VGG16 base is frozen during the initial training stage
- A custom classification head is added on top of the feature extractor
- Fine-tuning is optional and controlled in the configuration

## Screenshots Placeholder

Screenshots can be added to the documentation once the project is used on a real dataset.

## Results

Results are saved in the local project folders after training and evaluation. They include:

- model weights in [models](models)
- class names in [models/class_names.json](models/class_names.json)
- training history in [results/reports/training_history.json](results/reports/training_history.json)
- evaluation metrics in [results/reports/evaluation_metrics.json](results/reports/evaluation_metrics.json)
- confusion matrix in [results/plots/confusion_matrix.png](results/plots/confusion_matrix.png)
- training curves in [results/plots/training_history.png](results/plots/training_history.png)

## Limitations

- This is an academic prototype and not a clinical tool
- Model performance depends on dataset quality and class balance
- Training on a small dataset may underperform on unseen data
- Real-world medical deployment requires regulatory approval and clinical validation

## Future Enhancements

- Add more robust dataset curation
- Support explainability methods such as Grad-CAM
- Extend experimentation to other base models
- Add model versioning and experiment tracking
- Improve preprocessing and augmentation for real clinical imaging workflows

## Medical Disclaimer

This system is developed for academic and research purposes only. It is not a medical diagnostic tool and should not replace evaluation by a qualified healthcare professional.

## RetinaNet Clarification

RetinaNet is primarily an object-detection model and requires bounding-box annotations. This project focuses on chest X-ray image classification using VGG16 rather than object detection. A RetinaNet-based workflow would require a very different dataset and architecture and is not used here as a fake classifier.
