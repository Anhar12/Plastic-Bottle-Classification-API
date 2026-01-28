from flask import Flask, request, jsonify, send_from_directory, render_template
import numpy as np
import cv2
import tensorflow as tf
import os
import gdown
import uuid
import mysql.connector
import json
from urllib.parse import urlparse
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

app = Flask(__name__)

VOLUME_PATH = "/app/models"
UPLOAD_FOLDER = os.path.join(VOLUME_PATH, "uploads")
MODEL_PATH = os.path.join(VOLUME_PATH, "modelv6.h5")

os.makedirs(VOLUME_PATH, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

file_id = "1DUd--sm9-6SM9IbkyGg2HtIyaL1D-HdP"
model_url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(MODEL_PATH):
    gdown.download(model_url, MODEL_PATH, quiet=False, use_cookies=True)

model = None
predict_fn = None

CLASS_INFO = {
    0: {"code": "A", "brand": "Vit", "size": "1500ml", "weight": 27},
    1: {"code": "B", "brand": "Le Minerale", "size": "1500ml", "weight": 29},
    2: {"code": "C", "brand": "Sijiro", "size": "1500ml", "weight": 29},
    3: {"code": "D", "brand": "Squades", "size": "1500ml", "weight": 30},
    4: {"code": "E", "brand": "Crystaline", "size": "1500ml", "weight": 30},
    5: {"code": "F", "brand": "Aqua", "size": "1500ml", "weight": 30},
    6: {"code": "G", "brand": "Ron88", "size": "600ml", "weight": 16},
    7: {"code": "H", "brand": "Le Minerale", "size": "600ml", "weight": 17},
    8: {"code": "I", "brand": "Aqua", "size": "600ml", "weight": 17},
    9: {"code": "J", "brand": "Crystaline", "size": "600ml", "weight": 17},
    10: {"code": "K", "brand": "Pristine", "size": "600ml", "weight": 23},
}

def get_model():
    global model, predict_fn
    if model is None:
        m = tf.keras.models.load_model(MODEL_PATH)

        @tf.function(experimental_relax_shapes=True)
        def _predict(x):
            return m(x, training=False)

        model = m
        predict_fn = _predict

    return model, predict_fn

def get_db():
    mysql_url = os.getenv("MYSQL_PUBLIC_URL")
    if not mysql_url:
        raise Exception("MYSQL_PUBLIC_URL not set")

    url = urlparse(mysql_url)
    return mysql.connector.connect(
        host=url.hostname,
        user=url.username,
        password=url.password,
        database=url.path.lstrip("/"),
        port=url.port or 3306,
    )

def save_prediction_to_db(filename, code, brand, size, weight, confidence, preds):
    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO predictions
        (filename, code, brand, size, weight, confidence, preds)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (filename, code, brand, size, weight, confidence, preds)
    )

    conn.commit()
    cur.close()
    conn.close()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Missing file"}), 400

        file = request.files["file"]

        # Decode image from memory
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return jsonify({"error": "Invalid image"}), 400

        # Save image after decode
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4()) + ".jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.seek(0)
        file.save(filepath)

        # Preprocess
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
        img_rgb = img_rgb.astype("float32") / 255.0
        x = np.expand_dims(img_rgb, axis=0)

        _, pred_fn = get_model()
        preds = pred_fn(tf.convert_to_tensor(x)).numpy()[0]

        class_idx = int(np.argmax(preds))
        if class_idx not in CLASS_INFO:
            return jsonify({"error": "Invalid prediction output"}), 500

        confidence = float(np.max(preds)) * 100
        info = CLASS_INFO[class_idx]

        preds_json = json.dumps({
            CLASS_INFO[i]["code"]: float(preds[i])
            for i in range(len(preds))
        })

        save_prediction_to_db(
            filename=filename,
            code=info["code"],
            brand=info["brand"],
            size=info["size"],
            weight=info["weight"],
            confidence=confidence,
            preds=preds_json
        )

        return jsonify({
            "code": info["code"],
            "brand": info["brand"],
            "size": info["size"],
            "weight": info["weight"],
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/records", methods=["GET"])
def list_records():
    conn = get_db()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(rows)

@app.route("/images", methods=["GET"])
def list_images():
    files = sorted(os.listdir(UPLOAD_FOLDER), reverse=True)
    return jsonify(files)

@app.route("/images/<path:filename>", methods=["GET"])
def get_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/reset", methods=["DELETE"])
def reset_all():
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("DELETE FROM predictions")
        conn.commit()
        cur.close()
        conn.close()

        for f in os.listdir(UPLOAD_FOLDER):
            fp = os.path.join(UPLOAD_FOLDER, f)
            if os.path.isfile(fp):
                os.remove(fp)

        return jsonify({"status": "OK", "message": "Database and files cleared"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, threaded=False)
