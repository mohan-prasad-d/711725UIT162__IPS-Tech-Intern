from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import base64
import json
import os
import webbrowser

import cv2
import numpy as np

try:
    from tf_keras.layers import DepthwiseConv2D
    from tf_keras.models import load_model
    KERAS_LOADER = "tf_keras"
except Exception as tf_keras_error:
    try:
        from tensorflow.keras.layers import DepthwiseConv2D
        from tensorflow.keras.models import load_model
        KERAS_LOADER = "tensorflow.keras"
    except Exception:
        DepthwiseConv2D = None
        load_model = None
        KERAS_LOADER = "none"

HOST = "127.0.0.1"
PORT = 8000
ROOT = Path(__file__).resolve().parent
VALID_MOVES = {"stone", "paper", "scissor"}
MIN_CONFIDENCE = 0.50

model = None
class_names = []
model_error = ""

def patch_depthwise_conv2d_loader():
    if DepthwiseConv2D is None or getattr(DepthwiseConv2D, "_stone_game_patched", False):
        return

    original_from_config = DepthwiseConv2D.from_config

    def compatible_from_config(cls, config):
        config = dict(config)
        config.pop("groups", None)
        return original_from_config(config)

    DepthwiseConv2D.from_config = classmethod(compatible_from_config)
    DepthwiseConv2D._stone_game_patched = True

def clean_label(label):
    label = label.strip().lower()
    label = label.split(" ", 1)[1] if " " in label else label
    label = label.replace("scissors", "scissor").replace("sccsiors", "scissor")
    label = label.replace("rock", "stone")
    return label

def load_game_model():
    global model, class_names, model_error

    if load_model is None:
        model_error = "TensorFlow import failed"
        print("TensorFlow not available.")
        return

    model_path = ROOT / "keras_model.h5"
    labels_path = ROOT / "labels.txt"
    if not model_path.exists() or not labels_path.exists():
        model_error = "keras_model.h5 or labels.txt missing"
        print("keras_model.h5 or labels.txt missing.")
        return

    try:
        patch_depthwise_conv2d_loader()
        model = load_model(str(model_path), compile=False)
        class_names = [
            clean_label(line)
            for line in labels_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        # Warmup model graph execution
        warmup_batch = np.zeros((2, 224, 224, 3), dtype=np.float32)
        model(warmup_batch, training=False)
        model_error = ""
        print(f"Loaded model with labels: {', '.join(class_names)}")
    except Exception as exc:
        model = None
        class_names = []
        model_error = str(exc)
        print(f"Model load failed: {exc}")

def model_status():
    return {
        "loaded": model is not None,
        "labels": class_names,
        "error": model_error,
        "mode": "model" if model is not None else "fallback",
        "loader": KERAS_LOADER
    }

def decode_frame(data_url):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    image_bytes = base64.b64decode(data_url)
    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode camera frame.")
    return frame

def prepare_square_crop(crop):
    """Kai shape amungama irukka aspect ratio maarama center square crop panni 224x224 resize pannum"""
    if crop.size == 0:
        return np.zeros((224, 224, 3), dtype=np.float32)
    
    h, w = crop.shape[:2]
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    square_crop = crop[start_y:start_y+min_dim, start_x:start_x+min_dim]
    
    resized = cv2.resize(square_crop, (224, 224), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 127.5) - 1.0

def decide_winner(p1_res, p2_res):
    p1 = p1_res["move"]
    p2 = p2_res["move"]

    if p1 not in VALID_MOVES or p2 not in VALID_MOVES:
        return {
            "result": "waiting",
            "title": "Show both hands",
            "message": "Player 1 left side, Player 2 right side."
        }

    if p1_res["confidence"] < MIN_CONFIDENCE or p2_res["confidence"] < MIN_CONFIDENCE:
        return {
            "result": "waiting",
            "title": "Hold clearly",
            "message": "Move hands closer and keep inside camera box."
        }

    if p1 == p2:
        return {
            "result": "draw",
            "title": "Draw Round",
            "message": f"Both showed {p1.title()}."
        }

    p1_wins = (
        (p1 == "stone" and p2 == "scissor")
        or (p1 == "paper" and p2 == "stone")
        or (p1 == "scissor" and p2 == "paper")
    )

    if p1_wins:
        return {
            "result": "p1",
            "title": "Player 1 Wins",
            "message": f"{p1.title()} beats {p2.title()}."
        }

    return {
        "result": "p2",
        "title": "Player 2 Wins",
        "message": f"{p2.title()} beats {p1.title()}."
    }

def predict_frame(data_url):
    frame = decode_frame(data_url)
    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]
    middle = width // 2

    crop1 = frame[:, :middle]
    crop2 = frame[:, middle:]

    if model is None:
        p1_res = {"move": "none", "confidence": 0.0, "source": "fallback", "foundHand": False, "scores": {}}
        p2_res = {"move": "none", "confidence": 0.0, "source": "fallback", "foundHand": False, "scores": {}}
    else:
        # Fast Tensor Batch Execution (Single Pass for both players)
        img1 = prepare_square_crop(crop1)
        img2 = prepare_square_crop(crop2)
        batch = np.array([img1, img2], dtype=np.float32)

        predictions = model(batch, training=False).numpy()

        def parse_prediction(pred):
            idx = int(np.argmax(pred))
            conf = round(float(pred[idx]), 2)
            label = class_names[idx] if idx < len(class_names) else "none"
            if label not in VALID_MOVES:
                label = "none"
            return {
                "move": label,
                "confidence": conf,
                "source": "model",
                "foundHand": True,
                "scores": {class_names[i]: round(float(s), 2) for i, s in enumerate(pred) if i < len(class_names)}
            }

        p1_res = parse_prediction(predictions[0])
        p2_res = parse_prediction(predictions[1])

    winner = decide_winner(p1_res, p2_res)

    return {
        "playerOne": p1_res,
        "playerTwo": p2_res,
        "winner": winner
    }

class AppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_POST(self):
        if self.path != "/predict":
            self.send_error(404, "Endpoint not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body)
            result = predict_frame(payload.get("image", ""))
            self.send_json(result)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)

    def do_GET(self):
        if self.path == "/status":
            self.send_json(model_status())
            return
        super().do_GET()

    def send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

def run_server():
    load_game_model()

    server = None
    for port in range(PORT, PORT + 20):
        try:
            server = ThreadingHTTPServer((HOST, port), AppHandler)
            break
        except OSError:
            continue

    if server is None:
        raise RuntimeError("No free local port found.")

    port = server.server_address[1]
    url = f"http://{HOST}:{port}/index.html"
    print(f"Stone Paper Scissor server running at {url}")
    if os.environ.get("NO_BROWSER") != "1":
        webbrowser.open(url)
    server.serve_forever()

if __name__ == "__main__":
    run_server()