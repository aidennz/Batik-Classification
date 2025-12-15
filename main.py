import streamlit as st
import tensorflow as tf
import numpy as np
import base64
from PIL import Image

# =========================
# CONFIG
# =========================
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="Batik Classification",
    page_icon="🧵",
    layout="centered"
)

def set_background_local(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
        }}

        .block-container {{
            background-color: rgba(255, 248, 220, 0.9);
            padding: 2rem;
            border-radius: 12px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
set_background_local("./assets/old_paper.jpg")

# =========================
# LOAD MODEL & CLASS NAMES
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("batik_classification_model.keras")

@st.cache_resource
def load_class_names():
    return np.load("class_names.npy")

model = load_model()
class_names = load_class_names()

# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image)

    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image = tf.keras.applications.resnet50.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# =========================
# UI
# =========================
st.title("🧵 Batik Motif Classification")
st.markdown(
    """
    Aplikasi ini menggunakan **Deep Learning (ResNet50)**  
    untuk mengklasifikasikan **20 jenis motif batik Indonesia**.
    """
)

uploaded_file = st.file_uploader(
    "Upload gambar batik",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Gambar Input", use_container_width=True)

    if st.button("🔍 Prediksi"):
        with st.spinner("Menganalisis motif batik..."):
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)

            predicted_index = np.argmax(predictions)
            confidence = np.max(predictions) * 100

        st.success("Prediksi selesai!")

        st.markdown(f"""
        ### 🧠 Hasil Prediksi
        **Motif Batik:** `{class_names[predicted_index]}`  
        **Confidence:** `{confidence:.2f}%`
        """)

        # Top-3 prediction
        st.markdown("### 🔝 Top-3 Prediksi")
        top_3 = np.argsort(predictions[0])[::-1][:3]

        for i in top_3:
            st.write(f"- {class_names[i]} : {predictions[0][i]*100:.2f}%")
