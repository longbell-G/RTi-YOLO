import os

def find_labels_with_images(label_dir, target_classes):
    """
    Find label files containing the specified classes and their corresponding image files.
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    found_pairs = []

    for root, dirs, files in os.walk(label_dir):
        for file in files:
            if not file.endswith('.txt'):
                continue

            file_path = os.path.join(root, file)

            with open(file_path, 'r') as f:
                lines = f.readlines()

            found_classes = set()
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                cls_id = int(parts[0])
                if cls_id in target_classes:
                    found_classes.add(cls_id)

            if found_classes:
                # Find the corresponding image file
                base_name = os.path.splitext(file)[0]
                img_found = False

                for ext in image_extensions:
                    img_path = os.path.join(root.replace('labels', 'images'), base_name + ext)
                    if os.path.exists(img_path):
                        found_pairs.append({
                            'label': file_path,
                            'image': img_path,
                            'classes': sorted(found_classes)
                        })
                        break

    return found_pairs


# Configure paths
train_label_dir = r'E:\datasets\solar_cell_EL_image\dataset_split\labels\train'
val_label_dir = r'E:\datasets\solar_cell_EL_image\dataset_split\labels\val'
test_label_dir = r'E:\datasets\solar_cell_EL_image\dataset_split\labels\test'

# Target classes to search for
target_classes = [5, 6, 7, 10]

print("=" * 70)
print("Searching for label files containing classes 5,6,7,10 and their corresponding images")
print("=" * 70)

for dataset_name, label_dir in [("Training set", train_label_dir),
                                ("Validation set", val_label_dir),
                                ("Test set", test_label_dir)]:
    print(f"\n【{dataset_name}】")
    pairs = find_labels_with_images(label_dir, target_classes)

    if pairs:
        for i, pair in enumerate(pairs, 1):
            print(f"{i}. Label: {pair['label']}")
            print(f"   Image: {pair['image']}")
            print(f"   Classes: {pair['classes']}")
            print()
        print(f"Found {len(pairs)} files")
    else:
        print("No files containing the target classes found.")

# Count totals
total_train = len(find_labels_with_images(train_label_dir, target_classes))
total_val = len(find_labels_with_images(val_label_dir, target_classes))
total_test = len(find_labels_with_images(test_label_dir, target_classes))

print("\n" + "=" * 70)
print(f"Total:")
print(f"  Training set: {total_train} files")
print(f"  Validation set: {total_val} files")
print(f"  Test set: {total_test} files")
print(f"  Overall: {total_train + total_val + total_test} files")