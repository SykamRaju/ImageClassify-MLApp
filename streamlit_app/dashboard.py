# streamlit_app/dashboard.py
import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Garbage Classifier", layout="centered")

st.title("🗑️ Garbage Classification using AI")
st.markdown("Upload an image of waste and this app will tell what type it is.")

uploaded_file = st.file_uploader("Upload a garbage image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://127.0.0.1:8000/predict", files=files)

        if response.status_code == 200:
            result = response.json()
            st.success(f"🔍 Predicted: **{result['label']}** ({result['confidence']*100:.1f}% confidence)")
        else:
            st.error("❌ Prediction failed.")
