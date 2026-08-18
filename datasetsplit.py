import os
import shutil
import random

# ==================== Set paths (modify here) ====================
# 1. Path to the original image folder
image_folder = "E:/datasets/solar_cell_EL_image/trainval/JPEGImages"  # Folder containing 4500 images
# Path to the label folder (YOLO format .txt files)
label_folder = "E:/datasets/solar_cell_EL_image/trainval/trainvallabel"  # Folder with YOLO format labels

# Output folder path
output_folder = "E:/datasets/solar_cell_EL_image/dataset_split"  # The split dataset will be saved here

# 4. Path to the class file (contains the actual class names)
class_file = "E:/datasets/solar_cell_EL_image/annotation_classes.txt"  # Class file

# ==================== Set split ratios ====================
train_ratio = 0.6  # Training set 60%
val_ratio = 0.2    # Validation set 20%
test_ratio = 0.2   # Test set 20%

# ==================== Start splitting ====================

# 1. Create output directory structure
print("Creating output directories...")
dirs = [
    f"{output_folder}/images/train",
    f"{output_folder}/images/val",
    f"{output_folder}/images/test",
    f"{output_folder}/labels/train",
    f"{output_folder}/labels/val",
    f"{output_folder}/labels/test"
]

for d in dirs:
    os.makedirs(d, exist_ok=True)

# 2. Get all image files
print("Searching for image files...")
image_files = []
for f in os.listdir(image_folder):
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
        image_files.append(f)

print(f"Found {len(image_files)} images")

# 3. Match images with label files
print("Matching images and labels...")
valid_pairs = []
for img_name in image_files:
    # Get the base name (without extension)
    base_name = os.path.splitext(img_name)[0]

    # Find the corresponding label file
    label_name = f"{base_name}.txt"
    label_path = os.path.join(label_folder, label_name)

    if os.path.exists(label_path):
        valid_pairs.append((img_name, label_name))
    else:
        print(f"Warning: {img_name} has no corresponding label file")

print(f"Valid image-label pairs: {len(valid_pairs)}")

if len(valid_pairs) == 0:
    print("No valid image-label pairs found, exiting")
    exit()

# 4. Shuffle randomly
print("Shuffling data randomly...")
random.shuffle(valid_pairs)

# 5. Calculate split sizes
total = len(valid_pairs)
train_end = int(total * train_ratio)
val_end = train_end + int(total * val_ratio)

train_pairs = valid_pairs[:train_end]
val_pairs = valid_pairs[train_end:val_end]
test_pairs = valid_pairs[val_end:]

print(f"\nDataset split results:")
print(f"Training set: {len(train_pairs)} samples")
print(f"Validation set: {len(val_pairs)} samples")
print(f"Test set: {len(test_pairs)} samples")

# 6. Copy files to corresponding directories
def copy_files(pairs, split_name):
    """Copy image and label files to the specified directory"""
    for img_name, label_name in pairs:
        # Copy image
        src_img = os.path.join(image_folder, img_name)
        dst_img = os.path.join(output_folder, "images", split_name, img_name)
        shutil.copy2(src_img, dst_img)

        # Copy label
        src_label = os.path.join(label_folder, label_name)
        dst_label = os.path.join(output_folder, "labels", split_name, label_name)
        shutil.copy2(src_label, dst_label)

print("\nCopying files...")
copy_files(train_pairs, "train")
copy_files(val_pairs, "val")
copy_files(test_pairs, "test")

# 7. Read and use the actual class names
print("Reading class file...")
if not os.path.exists(class_file):
    print(f"Error: Class file {class_file} does not exist!")
    exit(1)

# Read the actual class names
with open(class_file, 'r', encoding='utf-8') as f:
    classes = [line.strip() for line in f if line.strip()]

print(f"Read {len(classes)} actual classes:")
for i, class_name in enumerate(classes):
    print(f"  {i}: {class_name}")

# Copy the class file to the output directory
shutil.copy2(class_file, os.path.join(output_folder, "classes.txt"))
print(f"Copied class file to {output_folder}/classes.txt")

# 8. Create data.yaml configuration file
print("Creating data.yaml configuration file...")

yaml_content = f"""# Dataset configuration file
path: {os.path.abspath(output_folder)}  # Dataset root directory
train: images/train  # Training set
val: images/val      # Validation set
test: images/test    # Test set

# Class information
nc: {len(classes)}  # Number of classes
names: {classes}  # Specific class names
"""

yaml_path = os.path.join(output_folder, "data.yaml")
with open(yaml_path, 'w', encoding='utf-8') as f:
    f.write(yaml_content)

print(f"\n✅ Dataset splitting completed!")
print(f"Output directory: {output_folder}")
print(f"Configuration file: {yaml_path}")
print(f"Number of classes: {len(classes)}")
print(f"Class names: {classes}")
print(f"\nDirectory structure:")
print(f"{output_folder}/")
print("├── data.yaml")
print("├── classes.txt")
print("├── images/")
print("│   ├── train/")
print("│   ├── val/")
print("│   └── test/")
print("└── labels/")
print("    ├── train/")
print("    ├── val/")
print("    └── test/")