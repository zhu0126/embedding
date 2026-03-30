# server.py
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import threading
import os

hf_token = os.getenv("HF_TOKEN")
app = Flask(__name__)

model = None
model_lock = threading.Lock()  # 避免多線程同時載入模型

def get_model():
    global model
    with model_lock:
        if model is None:
            # 第一次呼叫才載入模型
            model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu", use_auth_token=hf_token)
            # model = SentenceTransformer('ZHU1107/my-embedding-model',use_auth_token=hf_token)
        return model

@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

@app.route("/embedding", methods=["POST"])
def get_embedding():
    try:
        data = request.get_json(force=True)  # 即使 header 沒設 json 也能解析
        text = data.get("text")
        
        # 支援 list 或單條文字
        if text is None:
            return jsonify({"error": "No text provided"}), 400

        model_instance = get_model()
        
        # 如果傳入 list 就一次返回多個向量
        if isinstance(text, list):
            if len(text) > max_batch:
                text = text[:max_batch]  # 截斷到最大 batch
            vectors = model_instance.encode(text).tolist()
            return jsonify({"embedding": vectors})
        elif isinstance(text, str):
            vector = model_instance.encode(text).tolist()
            return jsonify({"embedding": vector})
        else:
            return jsonify({"error": "Invalid type for text"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0",port=port,threaded=True)
