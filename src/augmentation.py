from __future__ import annotations

import tensorflow as tf


def build_training_augmentation() -> tf.keras.Sequential:
    """Create a light augmentation pipeline suitable for chest X-ray data."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.05),
            tf.keras.layers.RandomTranslation(height_factor=0.08, width_factor=0.08),
        ],
        name="xray_training_augmentation",
    )
