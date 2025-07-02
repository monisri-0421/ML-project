import tensorflow as tf
import numpy as np
from PIL import Image
import io

model = tf.keras.models.load_model("models/model.h5")

def predict_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize((224,224))
    img_array = np.expand_dims(np.array(img)/255.0, axis=0)
    prediction = model.predict(img_array)
    return "Dog" if prediction[0][0] > 0.5 else "Cat"