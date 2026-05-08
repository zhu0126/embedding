# Embedding Vector Server

## 專案簡介
此伺服器負責將使用者輸入的文字轉換為語意向量（Vector Embedding），
提供推薦系統進行語意相似度比對與商品檢索。

本模組於專題中負責：
- 對話文字向量化
- 商品描述向量化
- 提供語意搜尋基礎

---

## 使用技術
- flask
- torch
- transformers
- gunicorn
- optimum[onnxruntime]
- numpy

---

## 專案檔案說明

| 檔案名稱 | 功能 |
|---|---|
| server.py | 啟動 API Server |
| requirements.txt | Python 套件需求 |
| Dockerfile | Docker 容器設定 |

---

## 安裝方式

### 1. 安裝套件
```bash
pip install -r requirements.txt
```

### 2.啟動server
```python
python server.py
```

### 3.Docker 執行方式
```bash
docker build -t embedding-server .
docker run -p 8000:8000 embedding-server
```
