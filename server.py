from flask import Flask, request, jsonify
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import threading
import os
import torch
from functools import lru_cache

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
            model = ORTModelForFeatureExtraction.from_pretrained(
                ONNX_MODEL_DIR,
                provider="CPUExecutionProvider",
                session_options={
                    "intra_op_num_threads": 1,
                    "inter_op_num_threads": 1,
                    "graph_optimization_level": "ORT_ENABLE_ALL"
                }
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

@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

@app.route("/embedding", methods=["POST"])
def get_embedding():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
    
        data = request.get_json(force=True)
        texts = data.get("text")
        
        if texts is None:
            return jsonify({"error": "Missing 'text' field. Example: {\"text\": \"你的文字\"}"}), 400
        
        # 統一轉成 list 方便處理
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, list):
            if len(texts) > max_batch:
                texts = texts[:max_batch]
        else:
            return jsonify({"error": "text must be str or list of str"}), 400

        # 取得模型
        model_instance, tok = get_model()

        # Tokenize
        encoded = tok(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # 執行 inference
        with torch.no_grad():
            outputs = model_instance(**encoded)

        # Mean Pooling + L2 Normalize（跟原本 SentenceTransformer 行為一致）
        embeddings = mean_pooling(outputs, encoded["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        result = embeddings.tolist()

        # 如果輸入是單一字串，就回傳單一向量（保持跟原本 API 相容）
        if len(result) == 1 and isinstance(data.get("text"), str):
            return jsonify({"embedding": result[0]})
        else:
            return jsonify({"embedding": result})

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print("Error:", error_detail)   # 方便在 Render Logs 查看
        return jsonify({"error": str(e)}), 500
        
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
