import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 🌈 Apply stylish background and fonts
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f8ff;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3 {
        color: #00274d;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model
model = load_model('cvd_prediction_model.h5')

# Define class labels
class_names = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

# App title
st.title("🩺 Cardiovascular Disease Risk Prediction")
st.markdown("### 📤 Upload a Retinal Image")
st.markdown("This app uses a trained deep learning model to predict eye diseases from retinal images.")

# Image uploader
uploaded_file = st.file_uploader("Choose a retina image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="📷 Uploaded Retina Image", use_column_width=True)

    # Preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Output
    st.success(f"🔍 Prediction: **{predicted_class}**")
    st.info(f"📊 Confidence Score: **{confidence:.2f}%**")
