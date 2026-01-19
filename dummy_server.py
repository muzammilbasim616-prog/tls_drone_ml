from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/ingest", methods=["POST"])
def ingest():
    data = request.get_json()
    if data is None:
        return jsonify({"status": "no data"}), 400

    print("Data received")
    print("Keys:", data.keys())
    print("Label:", data.get("label"))
    print("Confidence:", data.get("confidence"))
    print("GPS:", data.get("lat"), data.get("lon"))
    print("Image size:", len(data.get("image", "")))

    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
