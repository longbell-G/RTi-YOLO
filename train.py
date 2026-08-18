# from ultralytics import YOLO
# model = YOLO("yolo11n.pt")  # Load pre-trained model
# result = model.train(data='./data3.yaml', epochs=300, imgsz=640, batch=16)

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
import torch
torch.use_deterministic_algorithms(False)

"""
from ultralytics.utils import LOGGER
import logging
LOGGER.setLevel(logging.WARNING)  # Suppress basic model loading info
"""

class CustomSaveTrainer(DetectionTrainer):
    """
    Custom trainer that saves the best model based on the mAP50 metric.
    """

    def validate(self):
        results = super().validate()  # Receive as-is, do not force unpack into two values

        # Safely extract metrics from results
        metrics = results[0] if isinstance(results, tuple) else results

        # Attempt to get mAP50 and update best_fitness
        if hasattr(metrics, 'box'):
            new_fitness = getattr(metrics.box, 'map50', None)
            if new_fitness is not None:
                if self.best_fitness is None or new_fitness > self.best_fitness:
                    self.best_fitness = new_fitness

        return results  # Return unchanged, preserve structure


def train_model():
    # Load the model
    model = YOLO('HAM-ELNet.yaml')  # Ensure the weight path is correct
    # model.info()  # Print model info to check for channel errors
    # model = YOLO('runs/detect/train3/weights/last.pt')  # Weights saved from previous training

    # Training parameters
    results = model.train(
        # trainer = CustomSaveTrainer,
        data='./PIVELAD.yaml',
        epochs=300,
        cls=0.5,          # Increase classification loss weight
        box=7.5,
        dfl=1.5,
        kobj=2.0,         # Increase background penalty
        scale=0.9,        # Reduce scaling augmentation
        mixup=0.1,        # Enable mixup augmentation
        copy_paste=0.1,   # Slightly enable copy-paste
        degrees=10,
        hsv_s=0.5,        # Reduce saturation distortion
        hsv_v=0.3,        # Reduce brightness distortion
        cos_lr=True,      # Enable cosine learning rate scheduling
        close_mosaic=30,
        lr0=0.0005,       # Lower initial learning rate
        imgsz=640,
        batch=8,
        workers=0,        # Recommended for Windows
        device=0          # Use GPU
    )
    return results


if __name__ == '__main__':
    # Required for Windows multiprocessing
    import torch
    import multiprocessing
    multiprocessing.freeze_support()
    torch.multiprocessing.set_start_method('spawn', force=True)

    train_model()