import logging
import base64
import binascii
from datetime import datetime
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backend_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- IN-MEMORY STORAGE ---
# Store last 50 detections. 
# In a real app, use a database (SQLite/PostgreSQL).
DETECTION_HISTORY = []

@app.route("/detections", methods=["POST", "GET"])
def handle_detections():
    if request.method == "GET":
        # Return history (most recent first)
        return jsonify(list(reversed(DETECTION_HISTORY))), 200

    if request.method == "POST":
        try:
            data = request.get_json()
            if not data:
                logger.warning("Received empty or invalid JSON payload")
                return jsonify({"status": "no data"}), 400

            # 1. Basic Key Validation
            required_keys = ["image_base64", "label", "confidence", "lat", "lon", "timestamp"]
            missing_keys = [k for k in required_keys if k not in data]
            if missing_keys:
                logger.error(f"Missing keys: {missing_keys}")
                return jsonify({"status": "missing keys", "missing": missing_keys}), 400

            # 2. Extract Data
            label = data.get("label")
            conf = data.get("confidence")
            lat = data.get("lat")
            lon = data.get("lon")
            ts = data.get("timestamp")
            img_b64 = data.get("image_base64")

            # 3. Validation: Image Decode
            try:
                # Check if we can decode it (basic validity check)
                _ = base64.b64decode(img_b64, validate=True)
            except binascii.Error:
                logger.error("Failed to decode image_base64 string")
                return jsonify({"status": "invalid image data"}), 400
            except Exception as e:
                logger.error(f"Unexpected error decoding image: {e}")
                return jsonify({"status": "image decode error"}), 500

            # 4. Storage & Log Success
            
            # Create a clean record for storage
            record = {
                "label": label,
                "confidence": conf,
                "lat": lat,
                "lon": lon,
                "timestamp": ts,
                "image_base64": img_b64,
                "id": len(DETECTION_HISTORY) + 1 # Simple ID
            }
            
            # Append to history
            DETECTION_HISTORY.append(record)
            
            # Keep only last 50 to prevent memory growing forever in this dummy server
            if len(DETECTION_HISTORY) > 50:
                DETECTION_HISTORY.pop(0)

            logger.info(f"ACCEPTED detection: {label} ({conf:.2f}) @ [{lat}, {lon}] time={ts}")
            
            return jsonify({"status": "ok", "id": record["id"]}), 200

        except Exception as e:
            logger.critical(f"Server crashed processing request: {e}", exc_info=True)
            return jsonify({"status": "server error"}), 500

if __name__ == "__main__":
    # HOST 0.0.0.0 is crucial for local network visibility
    print("Server starting on 0.0.0.0:8000...")
    app.run(host="0.0.0.0", port=8000)
