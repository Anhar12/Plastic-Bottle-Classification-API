from flask import Flask, request, jsonify
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import tensorflow as tf 
import os
import gdown

app = Flask(__name__)

# ==== Load model ====
volume_path = "/models"
os.makedirs(volume_path, exist_ok=True)  # pastikan folder ada
model_path = os.path.join(volume_path, "modelv1.h5")

# File Google Drive
file_id = "1fiG4tBfBLG6_WU_xUbI2k6ss93E901DX"
url = f"https://drive.google.com/uc?id={file_id}"

done_flag = os.path.join(volume_path, "download_done.txt")

# Hanya download jika file belum ada
if not os.path.exists(model_path):
    gdown.download(url, model_path, quiet=False, use_cookies=True)
    with open(done_flag, "w") as f:
        f.write("done")

model = tf.keras.models.load_model(model_path)

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
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # CLAHE untuk kontras
    # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # l, a, b = cv2.split(lab)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # l_clahe = clahe.apply(l)
    # lab_clahe = cv2.merge((l_clahe, a, b))
    # img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Canny edge
    edges = cv2.Canny(img, threshold1=50, threshold2=180)

    # Resize sesuai input model CNN
    img_resized = cv2.resize(edges, (244, 244))

    # Normalisasi
    img_normalized = img_resized.astype("float32") / 255.0

    # (H, W, 1) grayscale
    img_final = np.expand_dims(img_normalized, axis=-1)

    # Tambah batch dimensi (1, H, W, 1)
    img_final = np.expand_dims(img_final, axis=0)

    return img_final

@app.route("/test", methods=["GET"])
def test():
    if os.path.exists(model_path) and os.path.exists(done_flag):
        return jsonify({"message": "Model loaded successfully"}), 200
    else:
        return jsonify({"error": "Model file not found"}), 404
    
@app.route("/ls", methods=["GET"])
def ls_files():
    model_dir = "/models"  # path mount Volume
    if os.path.exists(model_dir):
        return {"files": os.listdir(model_dir)}
    else:
        return {"error": "Folder model tidak ada"}

# ==== Endpoint predict dengan file upload ====
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Missing file"}), 400

    file = request.files["file"]
    img_bytes = file.read()

    try:
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
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
