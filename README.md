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
- Python
- Flask
- Sentence Transformers
- Docker

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
```bash
pip install -r requirements.txt
