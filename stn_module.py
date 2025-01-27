import torch
import torch.nn as nn
import torch.nn.functional as F

class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        # Mạng con để dự đoán phép biến đổi affine
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Lớp fully connected để dự đoán ma trận affine
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Khởi tạo ma trận affine với ma trận đơn vị
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        # Dự đoán ma trận affine
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # Áp dụng phép biến đổi affine lên hình ảnh
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
