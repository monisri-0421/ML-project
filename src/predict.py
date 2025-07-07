import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# ✅ Dynamically build correct path to model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.h5")

model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    prediction = model.predict(img_array)
    return "Dog" if prediction[0][0] > 0.5 else "Cat"
