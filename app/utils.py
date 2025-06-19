# app/utils.py
from tensorflow.keras.preprocessing import image
import numpy as np

IMG_HEIGHT = 150
IMG_WIDTH = 150
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']  # Match your folders

def preprocess_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array
