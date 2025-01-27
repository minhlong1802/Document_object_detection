import matplotlib.pyplot as plt
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import torch
import cv2
from ultralytics import YOLO
import numpy as np

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Cfg.load_config_from_name('vgg_transformer')
config['device'] = device
detector = Predictor(config)

# Load YOLOv8 model and perform detection
yolo_model = YOLO(r'D:\NCKH\best.pt')
img_path = r'D:\NCKH\test1.png'
results = yolo_model.predict(source=img_path)

# Load image with OpenCV
img_cv2 = cv2.imread(img_path)
if img_cv2 is None:
    raise Exception(f"Error: OpenCV could not read the image at '{img_path}'.")

# Convert OpenCV image to RGB format for Pillow
img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(img_rgb)

# Define the classes you want to keep
desired_classes = [2,3,4,5,6,7,8]  # Replace with the class IDs you want to keep, e.g., [0, 1]

# Define the padding ratio (percentage) to extend the bbox
y_padding_ratio = 0.3  # 20% padding
x_padding_ratio=0.04

# Create a list to hold cropped images
cropped_images = []

# Extract bounding boxes and crop the image
for result in results:
    for bbox in result.boxes:
        class_id = int(bbox.cls)  # Get the class ID for the bounding box
        if class_id in desired_classes:
            # Extract coordinates from tensor and convert to integers
            bbox_coords = bbox.xyxy[0].tolist()  # Convert tensor to list
            x1, y1, x2, y2 = map(int, bbox_coords)  # Convert coordinates to integers

            # Calculate the width and height of the bounding box
            width = x2 - x1
            height = y2 - y1
            
            # Calculate padding in pixels

            padding_y = int(height * y_padding_ratio)
            padding_x=int(width*x_padding_ratio)
            
            # Extend the bounding box with padding
            y1 = max(y1 - padding_y, 0)
            x1=max(x1-padding_x,0)
            y2 = min(y2 + padding_y, img_pil.height)
            x2=min(x2+padding_x,img_pil.width)
            # Crop the image using PIL
            cropped_img = img_pil.crop((x1, y1, x2, y2))
            cropped_images.append((cropped_img, class_id))  # Store both image and class ID
            
            # Optionally, save the cropped image
            #cropped_img.save(f'cropped_{class_id}_{x1}_{y1}.jpg')

# Display and predict text from cropped images
for cropped_img, class_id in cropped_images:
    try:
        plt.imshow(cropped_img)
        plt.axis('off')
        plt.show()
        
        s = detector.predict(cropped_img)
        print(f'Class {class_id}: {s}')
    except Exception as e:
        print(f"An error occurred during prediction: {e}")