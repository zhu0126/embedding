# server.py
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import threading

app = Flask(__name__)
model = SentenceTransformer('ZHU1107/my-embedding-model') 

@app.route("/embedding", methods=["POST"])
def get_embedding():
    try:
        data = request.get_json(force=True)  # 即使 header 沒設 json 也能解析
        text = data.get("text")
        
        # 支援 list 或單條文字
        if text is None:
            return jsonify({"error": "No text provided"}), 400

        # 如果傳入 list 就一次返回多個向量
        if isinstance(text, list):
            vectors = model.encode(text).tolist()
            return jsonify({"embedding": vectors})
        elif isinstance(text, str):
            vector = model.encode(text).tolist()
            return jsonify({"embedding": vector})
        else:
            return jsonify({"error": "Invalid type for text"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8000,threaded=True)
