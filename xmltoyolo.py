import os, glob, xml.etree.ElementTree as ET

# Path settings
A_folder = "E:/datasets/solar_cell_EL_image/trainval/Annotations"  # Your XML folder
B_folder = "E:/datasets/solar_cell_EL_image/trainval/trainvallabel"  # Output folder
class_file = "E:/datasets/solar_cell_EL_image/annotation_classes.txt"  # Class file

# Read classes
with open(class_file) as f:
    classes = [line.strip() for line in f]

# Ensure output folder exists
os.makedirs(B_folder, exist_ok=True)

# Convert all XML files
for xml_path in glob.glob(f"{A_folder}/*.xml"):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Image dimensions
    w = int(root.find("size/width").text)
    h = int(root.find("size/height").text)

    # Convert each object
    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bbox = obj.find("bndbox")

        x1 = int(bbox.find("xmin").text)
        y1 = int(bbox.find("ymin").text)
        x2 = int(bbox.find("xmax").text)
        y2 = int(bbox.find("ymax").text)

        # YOLO format conversion
        xc = (x1 + x2) / 2 / w
        yc = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        # Class ID
        class_id = classes.index(name)

        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    # Save txt file
    txt_name = os.path.basename(xml_path).replace(".xml", ".txt")
    with open(f"{B_folder}/{txt_name}", "w") as f:
        f.write("\n".join(lines))

# Save class file
with open(f"{B_folder}/classes.txt", "w") as f:
    f.write("\n".join(classes))

print("Conversion completed!")