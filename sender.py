import requests
import base64

API_URL = "http://15.206.79.226:8000/detect"

def send_detection(image_path, label, confidence, lat, lon):

    with open(image_path, "rb") as img_file:
        image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    payload = {
        "image": image_base64,
        "label": label,
        "confidence": confidence,
        "lat": lat,
        "lon": lon
    }

    response = requests.post(API_URL, json=payload)

    print("Status Code:", response.status_code)
    print("Response:", response.text)


# TEST
send_detection(
    image_path="test.jpg",
    label="crack",
    confidence=0.91,
    lat=11.25,
    lon=75.78
)
