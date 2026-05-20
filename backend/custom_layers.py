"""Custom Keras layers used by the CoralGuard fusion model."""

from __future__ import annotations

import keras
from keras import layers
import tensorflow as tf


@keras.saving.register_keras_serializable(package="Custom")
class CastToFloat32(layers.Layer):
    """Casts tabular inputs to float32 (used with mixed_float16 training)."""

    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

    def get_config(self):
        return super().get_config()


def get_custom_objects() -> dict[str, type]:
    return {
        "CastToFloat32": CastToFloat32,
        "Custom>CastToFloat32": CastToFloat32,
    }
