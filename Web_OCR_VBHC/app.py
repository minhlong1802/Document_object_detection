from flask import Flask, request, render_template, send_from_directory
import os
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import torch
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# Setup model configurations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Cfg.load_config_from_name('vgg_transformer')
config['device'] = device
detector = Predictor(config)

UPLOAD_FOLDER = r'D:\NCKH\Web_OCR_VBHC\static\uploads'
CROPPED_FOLDER = r'D:\NCKH\Web_OCR_VBHC\static\cropped'
RESULT_FOLDER = r'D:\NCKH\Web_OCR_VBHC\static\results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CROPPED_FOLDER'] = CROPPED_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROPPED_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLOv8 model
yolo_model = YOLO(r'D:\NCKH\best.pt')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Save uploaded file
        file = request.files.get('file')
        if file:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(img_path)

            # Perform detection
            results = yolo_model.predict(source=img_path)

            # Save image with labels
            output_filename = file.filename.rsplit('.', 1)[0] + '_label.jpg'
            output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
            results[0].save(output_path)

            # Load image with OpenCV and convert to RGB format for Pillow
            img_cv2 = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            # Define the classes you want to keep
            desired_classes = [2, 3, 4, 5, 6, 7, 8]

            # Define the padding ratio (percentage) to extend the bbox
            y_padding_ratio = 0.3  # 30% padding
            x_padding_ratio = 0.04

            # Create a list to hold cropped images and their OCR results
            cropped_results = []

            # Extract bounding boxes and crop the image
            for result in results:
                for bbox in result.boxes:
                    class_id = int(bbox.cls)  # Get the class ID for the bounding box
                    if class_id in desired_classes:
                        # Extract coordinates from tensor and convert to integers
                        bbox_coords = bbox.xyxy[0].tolist()
                        x1, y1, x2, y2 = map(int, bbox_coords)

                        # Calculate padding in pixels
                        width = x2 - x1
                        height = y2 - y1
                        padding_y = int(height * y_padding_ratio)
                        padding_x = int(width * x_padding_ratio)

                        # Extend the bounding box with padding
                        y1 = max(y1 - padding_y, 0)
                        x1 = max(x1 - padding_x, 0)
                        y2 = min(y2 + padding_y, img_pil.height)
                        x2 = min(x2 + padding_x, img_pil.width)

                        # Crop the image using PIL
                        cropped_img = img_pil.crop((x1, y1, x2, y2))

                        # Save cropped image
                        cropped_img_path = os.path.join(app.config['CROPPED_FOLDER'], f'cropped_{class_id}_{x1}_{y1}.jpg')
                        cropped_img.save(cropped_img_path)

                        # Perform OCR prediction
                        ocr_text = detector.predict(cropped_img)
                        cropped_results.append((f'cropped_{class_id}_{x1}_{y1}.jpg', ocr_text))

            # Render template with cropped images, OCR results, and the labeled image
            return render_template('results.html', cropped_results=cropped_results, result_image=output_filename)

    return render_template('upload.html')

@app.route('/static/cropped/<filename>')
def cropped_image(filename):
    return send_from_directory(app.config['CROPPED_FOLDER'], filename)

@app.route('/static/results/<filename>')
def result_image(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
