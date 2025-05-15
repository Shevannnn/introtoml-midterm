# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# from pathlib import Path

# # Load the model
# model = tf.keras.models.load_model("midterm_v6_wsl2_edgar.h5")

# # Folder with images
# image_folder = Path("C:/Users/evane/Downloads/Tests/")

# # Valid image extensions
# valid_exts = {'.jpg', '.jpeg', '.png'}

# # Prediction function
# def process_and_predict(img_path):
#     img = image.load_img(img_path, target_size=(128, 128))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = img_array / 255.0  # normalize

#     prediction = model.predict(img_array, verbose=0)
#     label = "Sedan" if prediction[0][0] < 0.5 else "SUV/Pickup Trucks"
#     confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
#     return img_path.name, label, confidence * 100

# # Scan folder and predict on each image
# for img_path in image_folder.rglob("*"):
#     if img_path.suffix.lower() in valid_exts:
#         name, label, conf = process_and_predict(img_path)
#         print(f"{name} {label} ({conf:.2f}%)")

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.metrics import f1_score, precision_score, confusion_matrix,accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model
model = tf.keras.models.load_model(r"C:\Users\evane\models\midterms\midterm_v8.h5")

# Define base image folder
base_folder = r"C:\Users\evane\Downloads\Tests"

# Label mapping
label_map = {"Car": 0, "SUV": 1}

# Prepare data
y_true = []
y_pred = []

# Traverse subdirectories (paper/plastic)
for label_name, label_value in label_map.items():
    subfolder = os.path.join(base_folder, label_name)
    if not os.path.isdir(subfolder):
        continue
    for img_name in os.listdir(subfolder):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(subfolder, img_name)

        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array, verbose=0)
        pred_value = float(prediction[0][0])
        pred_label = 1 if pred_value >= 0.5 else 0

        label = "Sedan" if pred_label == 0 else "SUV"
        confidence = pred_value if pred_label == 1 else 1 - pred_value

        y_true.append(label_value)
        y_pred.append(pred_label)

        # Print result
        print(f"{img_name}: {label} ({confidence * 100:.2f}% confidence)")


accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=["Sedan", "SUV"])

print("Classification Report: ")
print(report)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Car", "SUV"], yticklabels=["Car", "SUV"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
