from ultralytics import YOLO
import torch

print("CUDA available:", torch.cuda.is_available())

model = YOLO("yolov8n.pt")

results = model(
    source="test_images",
    device=0,
    conf=0.4,
    save=False
)

for r in results:
    img_name = r.path.split("\\")[-1]

    if r.boxes is None:
        continue

    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(float, box.xyxy[0])

        print({
            "image": img_name,
            "class": model.names[cls_id],
            "confidence": round(conf, 3),
            "bbox": [x1, y1, x2, y2]
        })
