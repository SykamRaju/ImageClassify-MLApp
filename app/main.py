# app/main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import uvicorn
import numpy as np
from app.utils import preprocess_image, CLASS_NAMES
import os
import gdown
import io

app = FastAPI()

# CORS settings (allow frontend to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, use only your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Download and Load Model ===
MODEL_PATH = "app/model/garbage_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1oo8Qm-s45YXYF_uee1RdV19FdL8xlAjx"

if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print("⬇️ Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

print("✅ Loading model...")
model = load_model(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "Garbage Classification API is live 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image_stream = io.BytesIO(contents)

    img_array = preprocess_image(image_stream)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction[0])
    confidence = float(prediction[0][class_index])
    label = CLASS_NAMES[class_index]

    return {
        "label": label,
        "confidence": round(confidence, 2)
    }

# Optional: local dev
# if __name__ == "__main__":
#     uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
