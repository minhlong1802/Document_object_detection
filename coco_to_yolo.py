import json
import os

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def is_valid_bbox(bbox):
    # Check if all values in the bbox are non-negative and width/height are positive
    return all(v >= 0 for v in bbox) and bbox[2] > 0 and bbox[3] > 0

def convert_coco_to_yolo(json_path, output_dir, class_mapping, other_class_start_id):
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = {img['id']: img for img in data['images']}
    annotations = data['annotations']
    
    other_class_id = other_class_start_id
    image_annotations = {}
    images_with_labels = set()
    images_without_labels = []

    # Group annotations by image
    for ann in annotations:
        bbox = ann['bbox']
        if not is_valid_bbox(bbox):
            continue  # Skip invalid bbox

        # Skip bbox values that are too large or have negative coordinates
        if ann['bbox'][0] > 9000 or ann['bbox'][0] < 0:
            continue

        image_id = ann['image_id']
        img = images[image_id]
        img_width = img['width']
        img_height = img['height']
        
        original_category_id = ann['category_id']
        
        if original_category_id == 5:
            continue  # Skip textbox label (category_id 5)
        elif original_category_id == 4:
            ann_type = ann['attributes']['type']
            if ann_type in class_mapping:
                new_category_id = class_mapping[ann_type]
            else:
                class_mapping[ann_type] = other_class_id
                new_category_id = other_class_id
                other_class_id += 1
        else:
            continue  # Skip other category_ids
        
        yolo_bbox = convert((img_width, img_height), bbox)
        
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append((new_category_id, yolo_bbox))
        images_with_labels.add(image_id)

    # Write to files
    for image_id, ann_list in image_annotations.items():
        img = images[image_id]
        output_file = os.path.join(output_dir, f"{os.path.splitext(img['file_name'])[0]}.txt")
        with open(output_file, 'a', encoding='utf-8') as f_out:
            for new_category_id, yolo_bbox in ann_list:
                f_out.write(f"{new_category_id} {' '.join(map(str, yolo_bbox))}\n")

    # Find images without labels
    for img_id, img in images.items():
        if img_id not in images_with_labels:
            images_without_labels.append(img['file_name'])

# Mapping types
class_mapping = {
    'PhotoStamp': 0,  # Convert to category_id 0 (Stamp)
    'ReplicaStamp': 0,  # Convert to category_id 0 (Stamp)
    'ConfirmStamp': 0,  # Convert to category_id 0 (Stamp)
    'PhotoSignature': 1,  # Convert to category_id 1 (Signature)
    'RealSignature': 1,  # Convert to category_id 1 (Signature)
    # Other types will be assigned to classes starting from ID 2
}

# Usage
json_path = 'D:/NCKH/VBHC_Dataset/project_văn+bản+hành+chính-2024_05_22_04_44_23-coco+1.0_1/annotations/instances_default.json'
output_dir = 'D:/NCKH/yolo_labels'
convert_coco_to_yolo(json_path, output_dir, class_mapping, other_class_start_id=2)
print(class_mapping)
