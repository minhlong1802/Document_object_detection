from collections import Counter
import os
import matplotlib.pyplot as plt

# Từ điển ánh xạ class_id với tên các lớp
class_names = {
    '0': 'Stamp',
    '1': 'Signature',
    '2': 'DocDate',
    '3': 'Brief',
    '4': 'Signer',
    '5': 'SourceDocNo',
    '6': 'Notation',
    '7': 'SenderCode',
    '8': 'DocType'
}

def get_class_distribution(annotation_folder):
    class_counter = Counter()

    for txt_file in os.listdir(annotation_folder):
        with open(os.path.join(annotation_folder, txt_file), 'r') as f:
            for line in f.readlines():
                class_id = line.split()[0]
                # Thay class_id bằng tên lớp từ từ điển
                class_name = class_names.get(class_id, 'Unknown')
                class_counter[class_name] += 1
    return class_counter

annotation_folder = r'D:\NCKH\YoloDataset_split\YoloDataset_split\labels\train'
class_distribution = get_class_distribution(annotation_folder)
print("Class Distribution: ", class_distribution)

plt.bar(class_distribution.keys(), class_distribution.values())

# Tăng kích thước của các giá trị cụ thể trong cột và hàng (trục x và trục y)
plt.xticks(fontsize=20)  # Tăng kích thước nhãn trên trục x (tên các class)
plt.yticks(fontsize=20)  # Tăng kích thước nhãn trên trục y (giá trị số lượng)

plt.xlabel('Class ID',fontsize=20)  # Kích thước của tên trục x
plt.ylabel('Số lượng',fontsize=20)  # Kích thước của tên trục y
plt.title('Phân phối class trong tập train', fontsize=16)  # Kích thước của tiêu đề

plt.show()
