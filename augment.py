import os
import cv2
import numpy as np
import albumentations as A

# Đường dẫn tới thư mục chứa ảnh gốc và file annotation
image_folder = r'D:\NCKH\YoloDataset_split\YoloDataset_split\images\train'
annotation_folder = r'D:\NCKH\YoloDataset_split\YoloDataset_split\labels\train'
output_image_folder = r'D:\NCKH\Augument\images'
output_annotation_folder = r'D:\NCKH\Augument\labels'

# Tạo các thư mục đầu ra nếu chưa tồn tại
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_annotation_folder, exist_ok=True)

# Định nghĩa các phép biến đổi với nhiễu Gauss
transform = A.Compose([
    A.Rotate(limit=15, p=0.2),                # Xoay ngẫu nhiên
    A.Affine(translate_percent={"x": 0.1, "y": 0.1}, scale=(0.8, 1.2), rotate=0, p=0.5),  # Biến đổi affine
    A.ShiftScaleRotate(shift_limit=0.1, 
                       scale_limit=0.2, 
                       rotate_limit=0, 
                       border_mode=cv2.BORDER_CONSTANT, 
                       p=0.5),  # Nghiêng và co giãn
    A.Blur(blur_limit=3, p=0.5),              # Làm mờ
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),  # Thêm nhiễu Gaussian
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Hàm để augment ảnh và cập nhật annotation
def augment_image_and_annotation(image_path, annotation_path, output_image_path, output_annotation_path):
    # Đọc ảnh và annotation
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    with open(annotation_path, 'r') as f:
        lines = f.readlines()

    # Tạo danh sách bounding boxes cho các annotation
    bboxes = []
    class_labels = []

    for line in lines:
        class_id, x_center, y_center, w, h = line.strip().split()
        x_center, y_center, w, h = float(x_center), float(y_center), float(w), float(h)

        # Kiểm tra và chuẩn hóa tọa độ nếu cần
        if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < w <= 1 and 0 < h <= 1:
            bboxes.append([x_center, y_center, w, h])  # Lưu dưới dạng danh sách
            class_labels.append(class_id)
        else:
            print(f"Invalid bbox {class_id} {x_center} {y_center} {w} {h} in {annotation_path}")

    # Kiểm tra nếu không có bounding boxes hợp lệ
    if not bboxes:
        print(f"No valid bounding boxes found for image: {image_path}. Skipping.")
        return

    # Thực hiện augment
    try:
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    except Exception as e:
        print(f"Error during augmentation for image {image_path}: {e}")
        return

    augmented_image = augmented['image']
    transformed_bboxes = augmented['bboxes']

    # Lưu ảnh mới với tên có hậu tố _augmented
    output_image_path_augmented = output_image_path.replace('.jpg', '_augmented.jpg').replace('.png', '_augmented.png')
    cv2.imwrite(output_image_path_augmented, augmented_image)

    # Cập nhật annotation
    output_annotation_path_augmented = output_annotation_path.replace('.txt', '_augmented.txt')
    with open(output_annotation_path_augmented, 'w') as f:
        for bbox, class_id in zip(transformed_bboxes, class_labels):
            x_center_new, y_center_new, w_new, h_new = bbox

            # Chỉ ghi lại những bbox hợp lệ
            if 0 <= x_center_new <= 1 and 0 <= y_center_new <= 1 and 0 < w_new <= 1 and 0 < h_new <= 1:
                f.write(f"{class_id} {x_center_new} {y_center_new} {w_new} {h_new}\n")
            else:
                print(f"Discarding bbox with class {class_id} due to invalid values: {x_center_new}, {y_center_new}, {w_new}, {h_new}")

# Duyệt qua tất cả các ảnh và annotation để augment
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(image_folder, filename)
        annotation_path = os.path.join(annotation_folder, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        # Đường dẫn lưu ảnh và annotation mới
        output_image_path = os.path.join(output_image_folder, filename)
        output_annotation_path = os.path.join(output_annotation_folder, filename.replace('.jpg', '.txt').replace('.png', '.txt'))

        augment_image_and_annotation(image_path, annotation_path, output_image_path, output_annotation_path)
