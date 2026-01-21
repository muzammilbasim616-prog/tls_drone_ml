from ultralytics import YOLO
import time
import sys
import threading
import random
import cv2
import base64
import requests
import math
from collections import deque, Counter
from datetime import datetime

# --- GPS SYSTEM ---
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

# --- EVENT DETECTION LOGIC ---

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) in meters.
    """
    if lat1 is None or lat2 is None:
        return 0.0
        
    R = 6371000  # Radius of earth in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

def is_new_detection(label, lat, lon, confidence, last_sent):
    """
    Decides if we should send a new alert based on the 4 rules.
    """
    # 1. New label not seen recently
    if last_sent["label"] != label:
        return True

    # 2. Same label but time gap > 30 seconds
    if time.time() - last_sent["time"] > 30:
        return True

    # 3. Same label but confidence crosses a threshold (+0.1)
    # Check if last_sent["confidence"] is not None to avoid initial error
    if last_sent["confidence"] is not None and confidence > last_sent["confidence"] + 0.1:
        return True

    # 4. Same label but at a different GPS location (> 5 meters)
    dist = haversine(lat, lon, last_sent["lat"], last_sent["lon"])
    if dist > 5:
        return True

    return False

def main():
    model_path = "runs/detect/train4/weights/best.pt"

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print("Starting live detection...")

    # Start GPS
    gps_thread = threading.Thread(target=gps_reader, daemon=True)
    gps_thread.start()

    # Inference stream
    results = model.predict(source=0, stream=True, conf=0.4, verbose=False)

    # CONFIG
    BACKEND_URL = "http://192.168.1.56:8000/detections"
    HISTORY_LEN = 10     # Track last N frames
    MIN_FRAMES = 5       # Must appear in K frames
    CONF_THRESHOLD = 0.85 # Average confidence threshold

    # STATE VARIABLES (minimal)
    event_active = False
    last_event_time = 0
    confidence_buffer = deque(maxlen=MIN_FRAMES)
    no_detection_count = 0
    
    DISAPPEAR_FRAMES = 10   # frames with no detection
    COOLDOWN_SECONDS = 20

    try:
        for r in results:
            # 1. DETECT
            current_best = None
            if r.boxes and len(r.boxes) > 0:
                box = max(r.boxes, key=lambda b: float(b.conf[0]))
                cls_id = int(box.cls[0])
                cnf = float(box.conf[0])
                lbl = model.names[cls_id]
                current_best = (lbl, cnf)

            # STEP 1: CONFIRM THE EVENT (stable detection)
            if current_best and current_best[1] >= CONF_THRESHOLD:
                # We have a valid detection
                confidence_buffer.append(current_best)
                no_detection_count = 0
            else:
                # No valid detection this frame
                confidence_buffer.clear()
                no_detection_count += 1

            # STEP 2: FIRE ONCE (LOCK IT)
            if not event_active and len(confidence_buffer) == MIN_FRAMES:
                # Extract details from the most recent detection in buffer
                lbl, cnf = confidence_buffer[-1]
                
                # GPS Snapshot
                current_lat = float(gps_data["lat"])
                current_lon = float(gps_data["lon"])
                
                # Prepare Payload
                frame = r.orig_img
                success, buffer = cv2.imencode(".jpg", frame)
                
                if success:
                    image_base64 = base64.b64encode(buffer).decode("utf-8")
                    timestamp = datetime.now().isoformat()
                    
                    payload = {
                        "image_base64": image_base64,
                        "label": lbl,
                        "confidence": cnf,
                        "lat": current_lat,
                        "lon": current_lon,
                        "timestamp": timestamp
                    }
                    
                    print("\n----- NEW EVENT DETECTED -----")
                    print(f"Event: {lbl} (Conf: {cnf:.2f})")
                    print(f"Location: {current_lat:.4f}, {current_lon:.4f}")
                    
                    try:
                        resp = requests.post(BACKEND_URL, json=payload, timeout=3)
                        if resp.status_code == 200:
                            print("✅ Alert Sent! Locking event...")
                            event_active = True
                            last_event_time = time.time()
                            # Clear buffer to prevent immediate re-trigger if logic changes
                            confidence_buffer.clear() 
                        else:
                            print(f"⚠️ Server exceeded: {resp.status_code}")
                    except Exception as e:
                        print(f"❌ Send failed: {e}")

            # STEP 3: UNLOCK ONLY WHEN IT’S GONE
            if event_active:
                if no_detection_count >= DISAPPEAR_FRAMES:
                    print("\n[RESET] Event ended (object disappeared). Ready for new events.")
                    event_active = False
                    no_detection_count = 0
                
                # Optional safety net
                elif time.time() - last_event_time > COOLDOWN_SECONDS:
                    print("\n[RESET] Event ended (timeout). Ready for new events.")
                    event_active = False
                    no_detection_count = 0

            # Optional: Print status dot to show it's alive
            # print(".", end="", flush=True)

    except KeyboardInterrupt:
        print("\nStopping detection.")
    except Exception as e:
        print(f"\nRuntime error: {e}")

if __name__ == "__main__":
    main()
