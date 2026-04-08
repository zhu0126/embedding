from flask import Flask, request, jsonify
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import threading
import os
import torch
from functools import lru_cache
import onnxruntime as ort

app = Flask(__name__)

model = None
tokenizer = None
model_lock = threading.Lock()

ONNX_MODEL_DIR = "ZHU1107/my-embedding-onnx"

max_batch = 16   # Render 免費方案建議不要超過 16~24，避免一次吃太多記憶體
MAX_LENGTH = 256

torch.set_num_threads(1)

def get_model():
    global model, tokenizer
    with model_lock:
        if model is None:
            print("正在載入 ONNX 模型...")
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            model = ORTModelForFeatureExtraction.from_pretrained(
                ONNX_MODEL_DIR,
                session_options=sess_options
            )
            tokenizer = AutoTokenizer.from_pretrained(ONNX_MODEL_DIR,use_fast=True)
            print("ONNX 模型載入完成")
        return model, tokenizer

def mean_pooling(model_output, attention_mask):
    """Mean Pooling（大多數 embedding model 使用的）"""
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def normalize_text(t):
    return t.strip().lower()

@lru_cache(maxsize=1000)
def embed_single(text):
    model_instance, tok = get_model()

    encoded = tok(
        [text],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model_instance(**encoded)

    embeddings = mean_pooling(outputs, encoded["attention_mask"])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings[0].tolist()

@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

@app.route("/embedding", methods=["POST"])
def get_embedding():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
    
        data = request.get_json()
        texts = data.get("text")
        
        if texts is None:
            return jsonify({"error": "Missing 'text' field. Example: {\"text\": \"你的文字\"}"}), 400
        
        # 統一轉成 list 方便處理
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        elif isinstance(texts, list):
            single_input = False
            if len(texts) > max_batch:
                texts = texts[:max_batch]
        else:
            return jsonify({"error": "text must be str or list of str"}), 400

        texts = [f"query: {normalize_text(t)}" for t in texts]
        
        results = []

        for t in texts:
            results.append(embed_single(t))

        if single_input:
            return jsonify({"embedding": results[0]})
        else:
            return jsonify({"embedding": results})

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print("Error:", error_detail)   # 方便在 Render Logs 查看
        return jsonify({"error": str(e)}), 500
        
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
