from ultralytics import YOLO

mode1 = YOLO("best.pt")
results = mode1("E:/datasets/solar_cell_EL_image/dataset_split/images/test",save=True)
