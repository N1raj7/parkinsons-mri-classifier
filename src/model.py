"""
EfficientNetB0 transfer-learning model for binary MRI classification
(Normal vs Parkinson's), plus a small from-scratch CNN baseline for comparison.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_transfer_model(input_shape=(224, 224, 3), fine_tune_at=100):
    """
    EfficientNetB0 backbone, frozen up to `fine_tune_at`, custom classification head.
    """
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=input_shape
    )

    # Freeze all layers first, then unfreeze the top block for fine-tuning
    base.trainable = True
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="efficientnet_parkinsons")
    return model, base


def build_baseline_cnn(input_shape=(224, 224, 3)):
    """
    Small custom CNN for comparison against the transfer-learning model.
    Report both in your write-up: 'transfer learning improved accuracy
    from X% (baseline CNN) to Y% (EfficientNetB0)'.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),
    ], name="baseline_cnn")
    return model


def compile_model(model, lr=1e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model
