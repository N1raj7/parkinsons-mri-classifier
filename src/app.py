"""
Streamlit demo: upload an MRI slice, get a prediction + Grad-CAM++ overlay.
Run: streamlit run src/app.py
"""

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from dataset import IMG_SIZE
from gradcam import find_last_conv_layer, gradcam_plusplus, overlay_heatmap

st.set_page_config(page_title="Parkinson's MRI Classifier", layout="centered")
st.title("🧠 Parkinson's Disease Detection from Brain MRI")
st.caption("EfficientNetB0 transfer learning + Grad-CAM++ explainability")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/model.h5")


model = load_model()
layer_container, layer_name = find_last_conv_layer(model)

uploaded = st.file_uploader("Upload an MRI slice (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded scan", width=300)

    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized, dtype=np.float32)
    img_pre = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_batch = tf.expand_dims(img_pre, 0)

    pred = model.predict(img_batch)[0][0]
    label = "Parkinson's" if pred >= 0.5 else "Normal"
    confidence = pred if pred >= 0.5 else 1 - pred

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.1%}")

    if st.button("Show Grad-CAM++ explanation"):
        # save temp file since gradcam helpers read from disk
        img.save("_temp_upload.jpg")
        cam = gradcam_plusplus(model, img_batch, layer_container, layer_name)
        overlayed = overlay_heatmap("_temp_upload.jpg", cam)
        st.image(overlayed, caption="Grad-CAM++ — regions influencing the prediction", width=300)

st.markdown("---")
st.caption(
    "⚠️ Research/educational demo only — not a diagnostic tool. "
    "Not validated for clinical use."
)
