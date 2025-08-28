import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL.Image import Resampling
import time
from streamlit_drawable_canvas import st_canvas

from src.version_manager import VersionManager
from src import config

st.set_page_config(
    page_title="MNIST Classifier",
    page_icon="🧮",
    layout="wide",
)

st.title("🧮 MNIST Digit Classifier")
st.markdown("Draw a digit (0–9) and the model will predict what it is.")

vm = VersionManager()
versions = vm.get_available_versions()

if not versions:
    st.error("No model versions available. Train or create a model first.")
    st.stop()

version = st.sidebar.selectbox("Select Model Version", options=versions, index=len(versions) - 1)
st.sidebar.info(f"Currently using model version: {version}")

CANVAS_SIZE = 280
STROKE_WIDTH = 10

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=STROKE_WIDTH,
    stroke_color="white",
    background_color="black",
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    drawing_mode="freedraw",
    key="canvas",
)

predict_disabled = canvas_result.image_data is None

if st.button("🚀 Predict Digit", type="primary", use_container_width=True, disabled=predict_disabled):
    with st.spinner("🤖 Calling the model..."):
        try:
            img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
            small_img = img.resize((28, 28), Resampling.LANCZOS).convert("L")

            image_data = np.array(small_img) / 255.0
            image_data = np.expand_dims(image_data, axis=-1).astype("float32")  # (28,28,1)

            payload = {"instances": [image_data.tolist()]}
            base = config.TF_SERVING_URL.rstrip("/")
            url = f"{base}/versions/{version}:predict"

            response = requests.post(url, json=payload)
            response.raise_for_status()
            preds = response.json()["predictions"][0]

            predicted_digit = int(np.argmax(preds))

            st.success(f"Predicted Digit: **{predicted_digit}** 🎉")

            fig, ax = plt.subplots()
            ax.bar(range(10), preds)
            ax.set_xticks(range(10))
            ax.set_xlabel("Digit")
            ax.set_ylabel("Probability")
            ax.set_title("Prediction Probabilities")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
