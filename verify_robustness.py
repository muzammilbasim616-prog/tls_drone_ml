import requests
import time
import base64
import random
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "http://127.0.0.1:8000/detections"
# Note: Using 127.0.0.1 for verification script to ensure we hit local, 
# even though server binds to 0.0.0.0 and ML script uses 192.168.1.56.
# They map to the same machine.

def send_request(i):
    payload = {
        "image_base64": base64.b64encode(b"fake_image_data").decode("utf-8"),
        "label": f"test_obj_{i}",
        "confidence": 0.99,
        "lat": 11.25,
        "lon": 75.78,
        "timestamp": "2026-01-20T10:00:00"
    }
    try:
        resp = requests.post(BASE_URL, json=payload, timeout=2)
        return resp.status_code
    except Exception as e:
        return str(e)

def main():
    print(f"Starting robustness test: 50 requests to {BASE_URL}...")
    success_count = 0
    fail_count = 0
    
    start = time.time()
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(send_request, range(50)))
        
    for res in results:
        if res == 200:
            success_count += 1
        else:
            fail_count += 1
            print(f"Failed request: {res}")
            
    print(f"\nTest finished in {time.time() - start:.2f}s")
    print(f"Success: {success_count}/50")
    print(f"Failures: {fail_count}/50")

    if fail_count == 0:
        print("PASS: Backend handled load.")
    else:
        print("FAIL: Backend dropped requests.")

if __name__ == "__main__":
    time.sleep(2) # Give server time to start
    main()
