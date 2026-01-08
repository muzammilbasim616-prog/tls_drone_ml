from ultralytics import YOLO
import json
import os

model = YOLO("runs/detect/train4/weights/best.pt")

results = model(
    source="test_images",
    device=0,
    conf=0.4,
    save=False
)

output = []

for r in results:
    record = {
        "image": os.path.basename(r.path),
        "detections": []
    }

    if r.boxes is not None:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])

            record["detections"].append({
                "class": model.names[cls_id],
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2]
            })

    output.append(record)

with open("detections.json", "w") as f:
    json.dump(output, f, indent=2)

print("Saved detections.json")
