# 使用官方 Python 3.11
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 複製依賴檔案
COPY requirements.txt .

# 安裝依賴
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 複製專案
COPY . .

# 預先載入模型（減少第一次請求時的延遲和記憶體峰值）
RUN python -c "
from sentence_transformers import SentenceTransformer
import os
hf_token = os.getenv('HF_TOKEN')
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu', use_auth_token=hf_token)
print('Model preloaded successfully')
"

# 設定環境變數，讓 Flask 不在開發模式
ENV FLASK_ENV=production

# 暴露端口
EXPOSE 10000

CMD ["gunicorn", "--bind", "0.0.0.0:10000", "server:app", "--workers=1", "--threads=4", "--timeout=180"]
