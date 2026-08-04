"""
Streamlit demo: upload an MRI slice, get a prediction + Grad-CAM++ overlay.
Run:
    streamlit run src/app.py
"""

import sys
from pathlib import Path

# Allow imports from the current src directory
sys.path.append(str(Path(__file__).parent))

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from src.dataset import IMG_SIZE
from src.gradcam import (
    find_last_conv_layer,
    gradcam_plusplus,
    overlay_heatmap,
)

st.set_page_config(
    page_title="Parkinson's MRI Classifier",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 Parkinson's Disease Detection from Brain MRI")
st.caption("EfficientNetB0 Transfer Learning + Grad-CAM++ Explainability")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/model.h5")


model = load_model()
layer_container, layer_name = find_last_conv_layer(model)

uploaded = st.file_uploader(
    "Upload an MRI slice",
    type=["jpg", "jpeg", "png"],
)

if uploaded:

    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded MRI", width=300)

    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized, dtype=np.float32)

    img_pre = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_batch = tf.expand_dims(img_pre, 0)

    pred = model.predict(img_batch, verbose=0)[0][0]

    if pred >= 0.5:
        label = "Parkinson's"
        confidence = pred
    else:
        label = "Normal"
        confidence = 1 - pred

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.1%}")

    if st.button("Show Grad-CAM++ Explanation"):

        # Save temporary image for Grad-CAM++
        temp_path = "_temp_upload.jpg"
        img.save(temp_path)

        cam = gradcam_plusplus(
            model,
            img_batch,
            layer_container,
            layer_name,
        )

        overlay = overlay_heatmap(temp_path, cam)

        st.image(
            overlay,
            caption="Grad-CAM++ Heatmap",
            width=300,
        )

st.markdown("---")
st.caption(
    "⚠️ This application is intended for research and educational purposes only. "
    "It is not a medical diagnostic tool."
)