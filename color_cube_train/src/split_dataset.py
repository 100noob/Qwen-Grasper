# Split the source dataset into training, validation, and test sets (70:20:10)
import os
import random
import shutil

# Source data directories
img_src = "source_data/src_images"
lab_src = "source_data/src_labels"

# Destination directories
img_train = "dataset/images/train"
img_val   = "dataset/images/val"
img_test  = "dataset/images/test"

lab_train = "dataset/labels/train"
lab_val   = "dataset/labels/val"
lab_test  = "dataset/labels/test"

# Create directories if they don't exist
os.makedirs(img_train, exist_ok=True)
os.makedirs(img_val, exist_ok=True)
os.makedirs(img_test, exist_ok=True)

os.makedirs(lab_train, exist_ok=True)
os.makedirs(lab_val, exist_ok=True)
os.makedirs(lab_test, exist_ok=True)

# Get all image filenames (excluding extensions)
images = [f for f in os.listdir(img_src) if f.endswith(".jpg")]
random.shuffle(images)

# Calculate split points (70% train, 20% val, 10% test)
total = len(images)
split_train = int(total * 0.70)
split_val = int(total * 0.90)

train_imgs = images[:split_train]
val_imgs = images[split_train:split_val]
test_imgs = images[split_val:]

def move(img_list, img_dst, lab_dst):
    """Copy images and their corresponding labels to destination folders"""
    for img in img_list:
        name = os.path.splitext(img)[0]
        # Copy image file
        shutil.copy(
            os.path.join(img_src, img),
            os.path.join(img_dst, img)
        )
        # Copy corresponding label file
        shutil.copy(
            os.path.join(lab_src, name + ".txt"),
            os.path.join(lab_dst, name + ".txt")
        )