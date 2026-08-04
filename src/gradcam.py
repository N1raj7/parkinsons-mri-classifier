"""
Grad-CAM++ implementation for visualizing which brain regions drove the
model's prediction. Sharper/more localized than vanilla Grad-CAM, especially
useful when the discriminative region is small (as with subtle MRI changes).

Usage: python src/gradcam.py --model model.h5 --image path/to/scan.jpg
"""

import argparse

import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

from src.dataset import IMG_SIZE


def find_last_conv_layer(model):
    """EfficientNet is nested inside our functional model — search inside it too."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return layer, sub.name
        if isinstance(layer, tf.keras.layers.Conv2D):
            return model, layer.name
    raise ValueError("No Conv2D layer found.")


def gradcam_plusplus(model, img_array, layer_container, layer_name):
    grad_model = tf.keras.models.Model(
        inputs=layer_container.input,
        outputs=[layer_container.get_layer(layer_name).output, layer_container.output]
    )

    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape3:
                conv_out, preds = grad_model(img_array)
                loss = preds[:, 0]
                tape3.watch(conv_out)
            grads = tape3.gradient(loss, conv_out)
        grads2 = tape2.gradient(grads, conv_out)
    grads3 = tape1.gradient(grads2, conv_out)

    conv_out = conv_out[0]
    grads = grads[0]
    grads2 = grads2[0] if grads2 is not None else tf.zeros_like(grads)
    grads3 = grads3[0] if grads3 is not None else tf.zeros_like(grads)

    alpha_num = grads2
    alpha_denom = 2.0 * grads2 + tf.reduce_sum(conv_out * grads3, axis=(0, 1), keepdims=True)
    alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, tf.ones_like(alpha_denom))
    alphas = alpha_num / alpha_denom

    weights = tf.reduce_sum(alphas * tf.nn.relu(grads), axis=(0, 1))
    cam = tf.reduce_sum(weights * conv_out, axis=-1)
    cam = tf.nn.relu(cam).numpy()
    cam = cv2.resize(cam, IMG_SIZE)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def overlay_heatmap(img_path, cam, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)


def main(args):
    model = tf.keras.models.load_model(args.model)
    layer_container, layer_name = find_last_conv_layer(model)
    print(f"Using conv layer: {layer_name}")

    img = tf.io.read_file(args.image)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img_pre = tf.keras.applications.efficientnet.preprocess_input(img)
    img_array = tf.expand_dims(img_pre, 0)

    pred = model.predict(img_array)[0][0]
    label = "Parkinson's" if pred >= 0.5 else "Normal"
    print(f"Prediction: {label} (confidence: {pred:.3f})")

    cam = gradcam_plusplus(model, img_array, layer_container, layer_name)
    overlayed = overlay_heatmap(args.image, cam)

    plt.figure(figsize=(6, 6))
    plt.imshow(overlayed)
    plt.title(f"Grad-CAM++ — Predicted: {label} ({pred:.2f})")
    plt.axis("off")
    plt.savefig(args.output, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model.h5")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default="gradcam_output.png")
    args = parser.parse_args()
    main(args)
