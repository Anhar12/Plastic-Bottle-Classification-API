from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
import os
import gdown

# ====== ENV & TF thread limits (pasang ini paling atas) ======
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

app = Flask(__name__)

# ====== Model path & download ======
volume_path = "/app/models"
os.makedirs(volume_path, exist_ok=True)
model_path = os.path.join(volume_path, "modelv1.h5")

file_id = "11UKY0PlGvoSubD_X3LXWrEvVoehG8nBl"
url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(model_path):
    gdown.download(url, model_path, quiet=False, use_cookies=True)

model = None
predict_fn = None

def get_model():
    global model, predict_fn
    if model is None:
        m = tf.keras.models.load_model(model_path)
        # Siapkan predict function sekali biar tidak retrace tiap request
        @tf.function(experimental_relax_shapes=True)
        def _pred(x):
            return m(x, training=False)
        model = m
        predict_fn = _pred
    return model, predict_fn

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

bg_color = np.array([132,132,132])
tolerance = np.array([15, 15, 15])

def compress_image(img_bgr, max_size=800):
    h, w = img_bgr.shape[:2]
    scale = max(h, w) / max_size
    if scale > 1:  # resize hanya jika lebih besar dari max_size
        new_w, new_h = int(w / scale), int(h / scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_bgr

def preprocess_image_bgr(img_bgr: np.ndarray):
    img_bgr = compress_image(img_bgr, max_size=800)
    
    # CLAHE
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.25, tileGridSize=(16, 16))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Buat mask background
    lower = np.clip(bg_color - tolerance, 0, 255)
    upper = np.clip(bg_color + tolerance, 0, 255)
    mask_bg = cv2.inRange(img_clahe, lower, upper)

    # Perbesar mask sedikit biar nutup tepi objek
    mask_bg_dilated = cv2.dilate(mask_bg, np.ones((5,5), np.uint8))

    # Blur hanya di bagian tepi background
    blurred = img_clahe.copy()
    blurred_bg = cv2.GaussianBlur(img_clahe, (3,3), 0)
    blurred[mask_bg_dilated > 0] = blurred_bg[mask_bg_dilated > 0]

    # Canny edge
    edges = cv2.Canny(blurred, threshold1=50, threshold2=180)

    # Hapus edge yang masih di background murni
    edges[mask_bg > 0] = 0

    # Resize -> (244, 244, 1)
    img_resized = cv2.resize(edges, (244, 244))
    img_normalized = img_resized.astype("float32") / 255.0
    img_final = np.expand_dims(img_normalized, axis=-1)   # (244,244,1)
    img_final = np.expand_dims(img_final, axis=0)         # (1,244,244,1)
    return img_final

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Missing file"}), 400

        # Decode image from memory
        file_bytes = np.frombuffer(request.files["file"].read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({"error": "Invalid image"}), 400

        _, pred_fn = get_model()
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (244, 244))
        img_normalized = img_resized.astype("float32") / 255.0

        x = np.expand_dims(img_normalized, axis=0)

        preds = pred_fn(tf.convert_to_tensor(x))
        preds = preds.numpy()
        class_idx = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds)) * 100

        info = CLASS_INFO[class_idx]
        return jsonify({
            "code": info["code"],
            "brand": info["brand"],
            "size": info["size"],
            "weight": info["weight"],
            "confidence": round(confidence, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, threaded=False)
