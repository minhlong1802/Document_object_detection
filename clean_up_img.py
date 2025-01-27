import os

# Directories
images_dir = r"D:\NCKH\Augument\labels"
annotations_dir = r"D:\NCKH\Augument\images"

# Get list of annotation files (without extension)
annotations_files = {os.path.splitext(file)[0] for file in os.listdir(annotations_dir)}

# Iterate through images directory
for image_file in os.listdir(images_dir):
    # Get the base name of the image file (without extension)
    base_name, ext = os.path.splitext(image_file)
    
    # Check if there's a corresponding annotation file
    if base_name not in annotations_files:
        # Full path to the image file
        image_path = os.path.join(images_dir, image_file)
        
        # Delete the image file
        os.remove(image_path)
        print(f"Deleted: {image_file}")

print("Cleanup complete.")
