# server.py
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
model = SentenceTransformer('Desktop/project/data/my_bge_model') 

@app.route("/embedding", methods=["POST"])
def get_embedding():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    vector = model.encode(text).tolist() 
    return jsonify({"embedding": vector})

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8000)