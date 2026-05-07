# 使用官方 Python 3.11
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 複製依賴檔案
COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安裝依賴
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 複製專案
COPY . .

# 設定環境變數，讓 Flask 不在開發模式
ENV FLASK_ENV=production

# 暴露端口
EXPOSE 10000

CMD ["gunicorn", "--bind", "0.0.0.0:10000", "server:app", "--workers=1", "--threads=4", "--timeout=180"]
