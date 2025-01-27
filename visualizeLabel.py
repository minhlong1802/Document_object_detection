import os
import cv2
import matplotlib.pyplot as plt
import random

# Đường dẫn đến thư mục chứa ảnh và nhãn
image_folder = r'D:\NCKH\Augument\images'
label_folder = r'D:\NCKH\Augument\labels'
output_folder = r'D:\NCKH\Augument\visualized'

# Tạo thư mục lưu ảnh visualize nếu chưa tồn tại
os.makedirs(output_folder, exist_ok=True)

# Lặp qua tất cả các ảnh trong thư mục
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Đường dẫn đầy đủ đến ảnh và nhãn tương ứng
        image_path = os.path.join(image_folder, filename)
        label_path = os.path.join(label_folder, filename.replace('.jpg', '.txt').replace('.png', '.txt'))

        # Đọc ảnh
        image = cv2.imread(image_path)

        # Khởi tạo danh sách để lưu các nhãn
        labels = []

        # Đọc file nhãn
        with open(label_path, 'r') as file:
            for line in file:
                class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())
                labels.append((int(class_id), x_center, y_center, bbox_width, bbox_height))

        # Lấy kích thước ảnh
        height, width, _ = image.shape

        # Màu cho các bounding box của từng class_id
        colors = {}

        # Vẽ bounding boxes và hiển thị class_id
        for label in labels:
            class_id, x_center, y_center, bbox_width, bbox_height = label
            
            # Chuyển đổi từ tọa độ chuẩn hóa (YOLO) sang tọa độ tuyệt đối
            x_min = int((x_center - bbox_width / 2) * width)
            y_min = int((y_center - bbox_height / 2) * height)
            x_max = int((x_center + bbox_width / 2) * width)
            y_max = int((y_center + bbox_height / 2) * height)
            
            # Tạo màu ngẫu nhiên cho mỗi class_id nếu chưa được tạo
            if class_id not in colors:
                colors[class_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            color = colors[class_id]
            
            # Vẽ bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Hiển thị class_id bên trên bounding box
            label_text = f'Class {class_id}'
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x_min, y_min - text_height - baseline), 
                          (x_min + text_width, y_min), color, -1)
            cv2.putText(image, label_text, (x_min, y_min - baseline), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Chuyển đổi ảnh từ BGR sang RGB cho matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Lưu ảnh đã visualize
        output_image_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_image_path, image)

        # Hiển thị ảnh đã vẽ bounding box (nếu cần)
        # plt.figure(figsize=(10, 10))
        # plt.imshow(image_rgb)
        # plt.axis('off')  # Ẩn trục
        # plt.show()
