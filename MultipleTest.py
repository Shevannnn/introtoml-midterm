import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from pathlib import Path

# Load the model
model = tf.keras.models.load_model("midterm_v4.keras")

# Folder with images
image_folder = Path("C:/Users/21-0270c/Downloads/test/")

# Valid image extensions
valid_exts = {'.jpg', '.jpeg', '.png'}

# Prediction function
def process_and_predict(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize

    prediction = model.predict(img_array, verbose=0)
    label = "Sedan" if prediction[0][0] < 0.5 else "SUV/Pickup Trucks"
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
    return img_path.name, label, confidence * 100

# Scan folder and predict on each image
for img_path in image_folder.rglob("*"):
    if img_path.suffix.lower() in valid_exts:
        name, label, conf = process_and_predict(img_path)
        print(f"{name} {label} ({conf:.2f}%)")
