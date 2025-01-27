import os
import shutil
import random
from collections import defaultdict
import albumentations as A
import cv2

# Directories
images_dir = r"D:\NCKH\Augument\images"
annotations_dir = r"D:\NCKH\Augument\labels"

train_images_dir = r"D:\NCKH\YoloDataset_split\YoloDataset_split\images\train"
train_labels_dir = r"D:\NCKH\YoloDataset_split\YoloDataset_split\labels\train"
valid_images_dir = r"D:\NCKH\YoloDataset_split\YoloDataset_split\images\val"
valid_labels_dir = r"D:\NCKH\YoloDataset_split\YoloDataset_split\labels\val"

# Split ratio
split_ratio = 0.8  # 80% for training, 20% for validation

# Create directories if they don't exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(valid_images_dir, exist_ok=True)
os.makedirs(valid_labels_dir, exist_ok=True)

# List all files in the images and annotations directories
images_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
annotations_files = {os.path.splitext(f)[0] for f in os.listdir(annotations_dir) if f.endswith('.txt')}

# Ensure that every image file has a corresponding annotation
images_files = [f for f in images_files if os.path.splitext(f)[0] in annotations_files]

# Dictionary to hold the association between classes and files
class_to_files = defaultdict(set)

# Populate the dictionary with files grouped by classes
for img_file in images_files:
    annotation_file = os.path.splitext(img_file)[0] + '.txt'
    annotation_path = os.path.join(annotations_dir, annotation_file)
    
    with open(annotation_path, 'r') as f:
        classes_in_file = {line.split()[0] for line in f.readlines()}
    
    for class_id in classes_in_file:
        class_to_files[class_id].add(img_file)

# Count class frequencies
class_frequencies = defaultdict(int)

for img_file in images_files:
    annotation_file = os.path.splitext(img_file)[0] + '.txt'
    annotation_path = os.path.join(annotations_dir, annotation_file)

    with open(annotation_path, 'r') as f:
        for line in f:
            class_id = int(line.split()[0])
            class_frequencies[class_id] += 1

# Merge files across classes while maintaining class distribution
all_files = set()
for files in class_to_files.values():
    all_files.update(files)

# Convert set to list and shuffle the files
all_files = list(all_files)
random.shuffle(all_files)

# Split files into training and validation sets
split_index = int(len(all_files) * split_ratio)
train_files = all_files[:split_index]
valid_files = all_files[split_index:]

# Function to copy files with existence check
def copy_files(files, source_dir, target_dir, augment=False):
    for file in files:
        src_image_path = os.path.join(source_dir, file)
        tgt_image_path = os.path.join(target_dir, file)
        annotation_file = os.path.splitext(file)[0] + '.txt'
        src_annotation_path = os.path.join(annotations_dir, annotation_file)
        tgt_annotation_path = os.path.join(target_dir.replace('images', 'labels'), annotation_file)

        if os.path.exists(src_image_path) and os.path.exists(src_annotation_path):
            shutil.copy2(src_image_path, tgt_image_path)
            shutil.copy2(src_annotation_path, tgt_annotation_path)
        else:
            print(f"Warning: {file} or its annotation does not exist, skipping.")

# Copy files to their respective directories
copy_files(train_files, images_dir, train_images_dir)  # Apply augmentation to training data
copy_files(valid_files, images_dir, valid_images_dir)  # No augmentation for validation data

print("Dataset split and augmentation complete.")
