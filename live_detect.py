from ultralytics import YOLO
import time
import sys
import threading
import random
import cv2
import base64
import requests

gps_data = {
    "lat": 0.0,
    "lon": 0.0
}

def gps_reader():
    global gps_data
    while True:
        # Simulate GPS update
        gps_data["lat"] = 11.2500 + random.uniform(-0.0005, 0.0005)
        gps_data["lon"] = 75.7800 + random.uniform(-0.0005, 0.0005)
        time.sleep(1)  # GPS updates once per second

def main():
    model_path = "runs/detect/train4/weights/best.pt"

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print("Starting live detection...")

    gps_thread = threading.Thread(target=gps_reader, daemon=True)
    gps_thread.start()

    results = model.predict(source=0, stream=True, conf=0.4, verbose=False)

    last_print_time = 0
    print_interval = 2.0
    frame_count = 0

    try:
        for r in results:
            frame_count += 1
            current_time = time.time()

            if current_time - last_print_time >= print_interval:
                frame = r.orig_img
                frame_shape = frame.shape if frame is not None else "NO FRAME"

                # Encode image to JPEG
                success, buffer = cv2.imencode(".jpg", frame)

                if not success:
                    print("Image encoding failed")
                    continue

                # Convert to Base64
                image_base64 = base64.b64encode(buffer).decode("utf-8")

                # Default values
                label = "none"
                confidence = 0.0

                if r.boxes and len(r.boxes) > 0:
                    # Pick highest confidence box
                    box = max(r.boxes, key=lambda b: float(b.conf[0]))
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    confidence = float(box.conf[0])

                # TEMP GPS placeholder (correct structure)
                # TEMP GPS placeholder (correct structure)
                lat = gps_data["lat"]
                lon = gps_data["lon"]

                payload = {
                    "image": image_base64,
                    "label": label,
                    "confidence": confidence,
                    "lat": lat,
                    "lon": lon
                }

                print("\n----- ML CYCLE -----")
                print("Payload keys:", payload.keys())
                print("Payload size (approx bytes):", len(str(payload)))

                try:
                    response = requests.post(
                        "http://127.0.0.1:5000/ingest",
                        json=payload,
                        timeout=1
                    )
                    print("Server response:", response.status_code)
                except Exception as e:
                    print("Send failed:", e)

                print("--------------------")

                last_print_time = current_time

    except KeyboardInterrupt:
        print("\nStopping detection.")
    except Exception as e:
        print(f"\nRuntime error: {e}")

if __name__ == "__main__":
    main()
