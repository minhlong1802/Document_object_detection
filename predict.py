import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2

# Load YOLOv8 model
yolo_model = YOLO(r'C:\Users\Admin\Downloads\YOLOv8l_50epochTrain\best.pt')

# Perform detection on an image
img_path = r'D:\NCKH\Augument\images\21_2012_QD-TTg_139105_page_1_augmented.jpg'
results = yolo_model.predict(source=img_path,conf=0.4, iou=0.5, show=False)  # Set show=False to prevent automatic showing

# Convert the predicted image to a format that matplotlib can display
annotated_image = results[0].plot(labels=True)  # Use labels=True to automatically draw class names

# If you want to customize the bounding boxes and labels further
names = {
    0: "Stamp",
    1: "Signature",
    2: "DocDate",
    3: "Brief",
    4: "Signer",
    5: "SourceDocNo",
    6: "Notation",
    7: "SenderCode",
    8: "DocType"
}

# Plot the image and bounding boxes with class names
plt.imshow(annotated_image)
plt.axis('off')  # Hide axes

# Iterate over results to customize labels
for result in results:
    for box in result.boxes:
        class_id = int(box.cls)  # Get the class ID
        label = names[class_id]  # Get the class name
        x1, y1, x2, y2 = box.xyxy[0]  # Get bounding box coordinates
        cv2.putText(annotated_image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

plt.show()
