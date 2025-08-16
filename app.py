from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
import os
import requests
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ==== Load model CNN ====
model = load_model('bestmodel.h5')

# ==== Mapping kelas A-K ====
CLASS_INFO = {
    0: {"code": "A", "brand": "Vit", "size": "1500ml", "weight": 27},
    1: {"code": "B", "brand": "Le Minerale", "size": "1500ml", "weight": 29},
    2: {"code": "C", "brand": "Sijiro", "size": "1500ml", "weight": 29},
    3: {"code": "D", "brand": "Squades", "size": "1500ml", "weight": 30},
    4: {"code": "E", "brand": "Crystaline", "size": "1500ml", "weight": 30},
    5: {"code": "F", "brand": "Aqua", "size": "1500ml", "weight": 30},
    6: {"code": "G", "brand": "Prima", "size": "600ml", "weight": 16},
    7: {"code": "H", "brand": "Le Minerale", "size": "600ml", "weight": 17},
    8: {"code": "I", "brand": "Aqua", "size": "600ml", "weight": 17},
    9: {"code": "J", "brand": "Crystaline", "size": "600ml", "weight": 17},
    10: {"code": "K", "brand": "Pristine", "size": "600ml", "weight": 23},
}


# ==== Fungsi preprocessing ====
def preprocess_image_from_bytes(img_bytes):
    # Buka dengan PIL lalu convert ke array OpenCV
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # CLAHE untuk kontras
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Canny edge
    edges = cv2.Canny(img_clahe, threshold1=50, threshold2=180)

    # Resize sesuai input model CNN (contoh 128x128)
    img_resized = cv2.resize(edges, (128, 128))

    # Normalisasi [0,1]
    img_normalized = img_resized.astype("float32") / 255.0

    # Tambah dimensi (H, W, 1) karena grayscale
    img_final = np.expand_dims(img_normalized, axis=-1)

    # Tambah batch dimensi (1, H, W, 1)
    img_final = np.expand_dims(img_final, axis=0)

    return img_final


# ==== Endpoint predict dari Supabase URL ====
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Missing 'url' in JSON body"}), 400

    image_url = data["url"]

    try:
        # Ambil gambar dari Supabase bucket storage
        response = requests.get(image_url)
        response.raise_for_status()
        img_bytes = response.content
    except Exception as e:
        return jsonify({"error": f"Failed to download image: {str(e)}"}), 400

    # Preprocess
    processed_img = preprocess_image_from_bytes(img_bytes)

    # Prediksi CNN
    preds = model.predict(processed_img)
    class_idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))

    # Ambil info kelas
    info = CLASS_INFO[class_idx]

    return jsonify({
        "code": info["code"],
        "brand": info["brand"],
        "size": info["size"],
        "weight": info["weight"],
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True)