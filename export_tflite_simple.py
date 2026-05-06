from ultralytics import YOLO

if __name__ == "__main__":
    # Load model
    model = YOLO("runs/detect/train4/weights/best.pt")
    
    # Export to TFLite directly with safe defaults (simplifying to resolve opset/tf issues)
    # Using opset=12 which is generally safer for TFLite conversion
    # Not using half=True on the first try to maximize chances of structural conversion success
    model.export(format="tflite", opset=12)
    print("Export pipeline complete.")
