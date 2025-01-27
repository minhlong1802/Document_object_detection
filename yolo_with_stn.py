from ultralytics import YOLO  # Import mô hình YOLOv8 gốc
from stn_module import STN  # Import module STN đã tạo ở bước 1
import torch

class YOLOv8WithSTN(YOLO):
    def __init__(self, model_path='yolov8n.pt'):
        super().__init__(model_path)  # Gọi tới constructor của YOLOv8 gốc
        self.stn = STN()  # Thêm STN vào mô hình

    def forward(self, x, *args, **kwargs):
        # Áp dụng STN trước khi thực hiện phát hiện với YOLOv8
        x = self.stn(x)
        # Gọi hàm forward của mô hình YOLOv8 gốc
        return super().forward(x, *args, **kwargs)
