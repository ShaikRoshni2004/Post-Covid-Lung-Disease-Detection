from __future__ import annotations

import tensorflow as tf

from config.settings import INPUT_SHAPE, LEARNING_RATE, RANDOM_SEED


def build_vgg16_classifier(
    num_classes: int,
    learning_rate: float = LEARNING_RATE,
    dropout_rate: float = 0.4,
    freeze_base: bool = True,
) -> tf.keras.Model:
    """Build a VGG16 transfer-learning classifier for chest X-ray classification."""
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    base_model = tf.keras.applications.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=INPUT_SHAPE,
        name="vgg16_base",
    )
    base_model.trainable = not freeze_base

    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="xray_input")
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = tf.keras.layers.Dense(512, activation="relu", name="feature_head")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout_layer")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="vgg16_transfer_learning")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def fine_tune_vgg16(model: tf.keras.Model, fine_tune_at: int = 15, learning_rate: float = 1e-5) -> tf.keras.Model:
    """Unfreeze only the last few VGG16 layers for a second-stage fine-tuning pass."""
    if not hasattr(model, "get_layer"):
        raise TypeError("The provided model does not support fine tuning.")

    base_model = model.get_layer("vgg16_base")
    if base_model is None:
        raise ValueError("The VGG16 base model was not found.")

    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def build_custom_cnn(num_classes: int, learning_rate: float = LEARNING_RATE) -> tf.keras.Model:
    """Create a lightweight custom CNN baseline for academic comparison."""
    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="cnn_input")
    x = tf.keras.layers.Rescaling(scale=1.0 / 255.0)(inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="custom_cnn_baseline")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model
