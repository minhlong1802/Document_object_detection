# Document Object Detection

## Project Overview

The **Document Object Detection** project is an AI-powered solution designed to detect and classify key components in administrative documents. Using the **YOLOv8** object detection model, this system can identify various elements such as:
- Titles
- Dates
- Keywords (e.g., "brief")
- Source document numbers

This project focuses on automating document processing workflows by providing precise and efficient object detection capabilities.

---
## Example output

![image](https://github.com/user-attachments/assets/d5f9a64d-af5c-4cc8-b801-a0e87f4dd702)


## Features

### 1. Object Detection
- Identify and locate key components in administrative documents.
- High precision detection for text-based objects.

### 2. Flexible Model Integration
- Leverages the **YOLOv8** model for cutting-edge performance.
- Supports custom training for specific document layouts and components.

### 3. Image Preprocessing
- Handles skewed and distorted document images.
- Supports preprocessing techniques like rotation correction to enhance accuracy.

### 4. Dataset Augmentation
- Augments data with tilted, blurred, and misaligned images for robust training.
- Balances class distribution for improved model performance.

---

## Tech Stack

### Machine Learning
- **YOLOv8**: State-of-the-art object detection model.
- **PyTorch**: Framework for model training and inference.

### Preprocessing
- **OpenCV**: For image preprocessing and augmentation.
- **Spatial Transformer Networks (STN)**: To handle skewed and distorted images.

### Tools and Libraries
- **Google Colab**: For training and testing the model.
- **Matplotlib**: For visualizing model performance and results.
- **Pandas**: For managing and analyzing annotation data.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/minhlong1802/Document_object_detection.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Document_object_detection
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the YOLOv8 model weights and place them in the `weights` directory.

---

## Training the Model

1. Prepare your dataset:
   - Ensure your dataset follows the YOLO format with images and annotations.
   - Place your dataset in the `data/` directory.

2. Update the configuration file:
   - Modify `config.yaml` to point to your dataset path and model parameters.

3. Train the model:
   ```bash
   python train.py --config config.yaml
   ```

4. Monitor training performance:
   - Check training metrics such as precision, recall, and mAP.

---

## Inference

1. Run inference on a single image:
   ```bash
   python detect.py --source path_to_image.jpg --weights weights/best.pt
   ```

2. Run inference on a batch of images:
   ```bash
   python detect.py --source path_to_images/ --weights weights/best.pt
   ```

3. View results:
   - The detected images with bounding boxes will be saved in the `results/` folder.

---

## Dataset Augmentation

To improve the model's robustness, augment the dataset with:
- Rotated images
- Skewed images
- Blurred images
- Adjusted brightness and contrast

Use the `augmentation.py` script to apply these transformations:
```bash
python augmentation.py --input data/raw --output data/augmented
```

---

## Contribution

If you'd like to contribute to this project:
1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes and push to your branch:
   ```bash
   git commit -m "Add your message here"
   git push origin feature/your-feature-name
   ```
4. Create a pull request.

---

## Contact

For inquiries or support, feel free to contact me:
- **Email**: nguyenlong18022004@gmail.com
- **GitHub**: [minhlong1802](https://github.com/minhlong1802)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
