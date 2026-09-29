# Dataset Setup

This project expects a chest X-ray dataset containing actual medical images for classification.

## Required Dataset Type

Use a labeled chest X-ray dataset with image folders for each disease class.
The code automatically discovers class names from the training directory names.

## Required Directory Structure

```text
dataset/
    train/
        class_1/
        class_2/
        ...
    validation/
        class_1/
        class_2/
        ...
    test/
        class_1/
        class_2/
        ...
```

## Training, Validation, and Test Organization

- Each class must have its own folder under `dataset/train`
- The same class names should exist in `dataset/validation` and `dataset/test`
- All folders should contain real chest X-ray images
- Supported formats: `.jpg`, `.jpeg`, and `.png`

## Important Note

The dataset is intentionally not bundled in GitHub because medical imaging datasets can be large, may have licensing terms, and may require institutional access or data-use agreement approval.

## Class Example

Depending on the dataset, class folders may be named things like:

- COVID
- Normal
- Lung Opacity
- Viral Pneumonia

The application does not hard-code class names. It reads the folder labels automatically.

## Validation Rules

The project validates that:

- the dataset folder exists
- train/validation/test splits exist
- at least two class folders exist
- each class folder has valid images
- the class sets match across train/validation/test

If the dataset is missing or incomplete, the application will show a friendly message instead of crashing.

## Medical Research Warning

This dataset is for academic and research use only. The project is not a medical diagnosis system and does not claim to replace physician evaluation.
