import os
import shutil
import random

random.seed(42)

base_path = r"C:\Users\evane\.cache\kagglehub\datasets\ryanholbrook\car-or-truck\versions\1"
original_train_dir = os.path.join(base_path, "train")
new_base = os.path.join(base_path, "split_data")

# 80% train, 10% val, 10% test
splits = {"train": 0.8, "val": 0.1, "test": 0.1}

# Create new folders
for split in splits:
    for label in os.listdir(original_train_dir):
        os.makedirs(os.path.join(new_base, split, label), exist_ok=True)

# Split each class
for label in os.listdir(original_train_dir):
    label_path = os.path.join(original_train_dir, label)
    files = [f for f in os.listdir(label_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(files)

    total = len(files)
    train_end = int(total * splits["train"])
    val_end = train_end + int(total * splits["val"])

    split_files = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }

    for split, file_list in split_files.items():
        for file in file_list:
            src = os.path.join(label_path, file)
            dst = os.path.join(new_base, split, label, file)
            shutil.copy2(src, dst)

print("Split complete. New data is in:", new_base)
