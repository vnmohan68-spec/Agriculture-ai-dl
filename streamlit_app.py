import streamlit as st
import numpy as np
import json
from PIL import Image
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered",
)

# ── Load model & class labels ─────────────────────────────────────────────────
MODEL_PATH  = "plant_disease_model.keras"
CLASSES_PATH = "plant_classes.json"

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    # Import inside function to avoid slow startup when model isn't needed yet
    import tensorflow as tf
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_classes():
    if not os.path.exists(CLASSES_PATH):
        st.error(f"Classes file not found: {CLASSES_PATH}")
        st.stop()
    with open(CLASSES_PATH) as f:
        data = json.load(f)
    return data["classes"], data["img_size"]

model              = load_model()
class_names, IMG_SIZE = load_classes()

# ── Helper: preprocess image ──────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    """Resize, convert to RGB, normalise to [0, 1] and add batch dim."""
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr   = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)          # shape: (1, H, W, 3)

# ── Helper: format class label ────────────────────────────────────────────────
def format_label(label: str) -> str:
    return label.replace("_", " ").title()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🌿 Plant Disease Detector")
st.markdown(
    "Upload a leaf image and the model will identify the plant species "
    "and diagnose any disease."
)
st.divider()

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported plants: Bell Pepper, Potato, Tomato",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction")

        with st.spinner("Analysing…"):
            input_arr  = preprocess(image)
            predictions = model.predict(input_arr, verbose=0)[0]

        top_idx        = int(np.argmax(predictions))
        top_label      = class_names[top_idx]
        top_confidence = float(predictions[top_idx]) * 100

        # ── Result badge ──
        is_healthy = "healthy" in top_label.lower()
        badge_color = "🟢" if is_healthy else "🔴"

        st.markdown(f"### {badge_color} {format_label(top_label)}")
        st.metric("Confidence", f"{top_confidence:.1f}%")

        if is_healthy:
            st.success("The plant appears **healthy**! No disease detected.")
        else:
            st.warning(
                f"**Disease detected:** {format_label(top_label)}. "
                "Consider consulting an agricultural expert."
            )

        # ── Top-3 breakdown ──
        st.markdown("#### Top predictions")
        top3_idx = np.argsort(predictions)[::-1][:3]
        for idx in top3_idx:
            label = format_label(class_names[idx])
            conf  = float(predictions[idx]) * 100
            st.progress(conf / 100, text=f"{label}  —  {conf:.1f}%")

st.divider()
st.caption(
    "Supported classes: Bell Pepper (Bacterial Spot / Healthy) · "
    "Potato (Early Blight / Late Blight) · "
    "Tomato (Bacterial Spot / Early Blight / Late Blight / Leaf Mold / "
    "Septoria Leaf Spot / Target Spot / Mosaic Virus)"
)
