from ultralytics import YOLO

# Method 1: Use the resume parameter directly
model = YOLO('runs/detect/train10/weights/last.pt')  # Weights saved from the previous training

# Continue training
results = model.train(
    data='PIVELAD.yaml',
    epochs=300,      # Total number of epochs, not additional epochs
    imgsz=640,
    batch=8,
    resume=True,     # Key parameter, automatically resumes training
    name='detect',   # Use the same experiment name, will automatically continue logging
    #mixup=0.0,      # Disable MixUp (direct cause of crash)
    #copy_paste=0.0, # Also disable Copy-Paste
)