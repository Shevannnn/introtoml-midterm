import os
import platform
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the base directory
if platform.system() == 'Linux':
    base_dir = "/mnt/c/Users/evane/.cache/kagglehub/datasets/ryanholbrook/car-or-truck/versions/1/split_data"
    # base_dir = "/mnt/c/Users/evane/Downloads/Midterms/Midterms/final_dataset"
else:
    base_dir = r"C:\Users\evane\.cache\kagglehub\datasets\ryanholbrook\car-or-truck\versions\1\split_data"


# Define subdirectories
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# Image settings
img_size = (128, 128)
batch_size = 32

# Data generators
datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

test_gen = datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# CNN Model
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
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_gen, validation_data=val_gen, epochs=10)

def evaluate_model(model, test_gen):
    # Get true labels and predicted probabilities
    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen)
    
    # Convert probabilities to binary predictions (threshold = 0.5)
    y_pred = (y_pred_probs > 0.5).astype(int).reshape(-1)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=test_gen.class_indices.keys()))

    # Print F1, Precision, Recall separately
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    print(f"F1 Score: {f1:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_gen.class_indices.keys(), yticklabels=test_gen.class_indices.keys())
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

evaluate_model(model, test_gen)

# Save the model
model.save("midterm_v7.h5")
print("Training complete! Model saved.")

# Final evaluation on test set only
loss, accuracy = model.evaluate(test_gen)
print(f"Test Accuracy: {accuracy:.2f}, Loss: {loss:.4f}")
