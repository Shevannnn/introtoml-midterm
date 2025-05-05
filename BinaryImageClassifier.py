import os
import kagglehub
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

path = kagglehub.dataset_download("ryanholbrook/car-or-truck")
print("Path to dataset files:", path)

# Get image paths n shit
dataset_path = os.path.join(path, "train")

classes = ["car", "truck"]
image_paths = []
labels = []

for label, cls in enumerate(classes):
    class_dir = os.path.join(dataset_path, cls)
    for fname in os.listdir(class_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(class_dir, fname))
            labels.append(cls)

df = pd.DataFrame({
    'filename': image_paths,
    'class': labels
})

# split to 80/10/10, 80 train, 10 test and val
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['class'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['class'], random_state=42)
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# Image settings
img_size = (128, 128)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_dataframe(train_df, x_col='filename', y_col='class',
                                        target_size=img_size, class_mode='binary',
                                        batch_size=batch_size, shuffle=True)

val_gen = datagen.flow_from_dataframe(val_df, x_col='filename', y_col='class',
                                      target_size=img_size, class_mode='binary',
                                      batch_size=batch_size, shuffle=False)

test_gen = datagen.flow_from_dataframe(test_df, x_col='filename', y_col='class',
                                       target_size=img_size, class_mode='binary',
                                       batch_size=batch_size, shuffle=False)


# Enhanced CNN Model Structure
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),  # Reduces overfitting
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_gen, validation_data=val_gen, epochs=10)

model.save("midterm_v3.keras")

print("Training complete! Model saved.")

loss, accuracy = model.evaluate(test_gen)
print(f"Validation Accuracy: {accuracy:.2f}, Loss: {loss:.4f}")