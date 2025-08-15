# Breeze-ASR-25 微調管線
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/Luca0x5755/breeze-finetune-pipeline) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](/LICENSE)

> 一個專為 MediaTek-Research/Breeze-ASR-25 模型設計的本地化微調與評估管線，整合了語義相似度評估，提供比傳統 WER 更深入的模型性能洞察。

## 🚀 快速開始

```bash
# 1. 準備您的音訊資料與標註檔案
#    - 音訊檔放入 /standard 資料夾
#    - 建立 metadata_train_fixed.csv 與 metadata_test_fixed.csv

# 2. 啟動 Docker 容器
docker-compose up -d --build

# 3. 進入容器並開始訓練
docker-compose exec asr-training bash
python finetune_Breeze_local.py
```

## 📋 目錄
- [關於專案](#關於專案)
- [功能特色](#功能特色)
- [安裝指南](#安裝指南)
- [使用方式](#使用方式)
- [貢獻指南](#貢獻指南)
- [授權條款](#授權條款)
- [聯絡我們](#聯絡我們)

## 📖 關於專案

### 問題背景
傳統的語音辨識（ASR）模型評估主要依賴字詞錯誤率（Word Error Rate, WER），但 WER 無法完全反映模型對語義的理解程度。此外，針對特定領域（如台語）微調高效能 ASR 模型（如 Breeze-ASR-25）的過程需要一個標準化且易於複製的環境。

### 解決方案
本專案提供一個基於 Docker 的完整解決方案，用於在本機環境中微調 Breeze-ASR-25 模型。其核心特色是整合了`語義相似度`評估腳本，讓開發者不僅能追蹤字詞層面的準確度，更能評估模型生成的語句在語義上是否與參考文本相符，為模型優化提供更全面的視角。

### 技術棧
- **核心模型**: `MediaTek-Research/Breeze-ASR-25`
- **開發框架**: `PyTorch`, `Transformers`, `datasets`
- **評估工具**: `evaluate` (for WER), `sentence-transformers` (for Semantic Similarity)
- **執行環境**: `Docker`, `Docker Compose`

## ✨ 功能特色

- 🔥 **Docker化環境**: 使用 Docker 和 Docker Compose 快速建立一致的開發與訓練環境，解決依賴問題。
- 🧠 **語義相似度評估**: 內建 `evaluate_with_semantic_similarity.py`，使用向量模型計算預測文本與參考文本的語義相似度，提供超越傳統 WER 的評估指標。
- 🔧 **本地化訓練**: `finetune_Breeze_local.py` 腳本專為本地 GPU 環境優化，簡化了微調流程。
- 📁 **標準化目錄結構**: 專案結構清晰，僅需將音訊檔案放入 `/standard` 目錄即可開始訓練。

## 🔧 安裝指南

### 系統要求
- Docker >= 20.10
- Docker Compose
- NVIDIA GPU 並已安裝 NVIDIA Container Toolkit
- Git

### 安裝步驟
```bash
# 1. 複製專案
git clone https://github.com/your-username/breeze-finetune-pipeline.git

# 2. 進入目錄
cd breeze-finetune-pipeline

# 3. 準備資料
#    - 將所有 .wav 音訊檔案放入 ./standard/ 對應的子資料夾中。
#    - 準備訓練與測試用的 CSV 檔案：
#      - `metadata_train_fixed.csv`
#      - `metadata_test_fixed.csv`
#    - CSV 檔案需包含 `file` (音訊檔絕對路徑) 和 `中文意譯` (對應的標註文本) 兩個欄位。

# 4. 建立並啟動 Docker 容器
#    此命令會根據 Dockerfile 建立映像檔，並在背景啟動一個服務
docker-compose up -d --build
```

## 🎯 使用方式

### 1. 進入 Docker 容器
```bash
docker-compose exec asr-training bash
```

### 2. 執行微調訓練
在容器內執行以下命令來啟動訓練腳本。腳本會自動讀取 `metadata_train_fixed.csv` 和 `metadata_test_fixed.csv` 進行訓練。
```bash
python finetune_Breeze_local.py
```
訓練完成後，最佳模型將被保存在 `breeze-asr-25-local-hokkian_v1` 目錄下。

### 3. 執行語義相似度評估
評估腳本可以幫助您深入分析模型在測試集上的表現。
首先，您需要修改 `evaluate_with_semantic_similarity.py` 中的參數：
```python
# evaluate_with_semantic_similarity.py

# ...
if __name__ == "__main__":
    # --- 請修改以下參數 ---

    # 訓練好的模型路徑
    MY_FINE_TUNED_MODEL_PATH = "./breeze-asr-25-local-hokkian_v1" # 確認此路徑正確

    # 測試音訊檔案資料夾 (例如，您的測試集 .wav 檔案所在位置)
    TEST_FOLDER_PATH = "/standard/your_test_audio_folder"

    # 參考文本CSV檔案 (例如，您的測試集標註檔)
    REFERENCE_CSV = "metadata_test_fixed.csv"
# ...
```
修改完成後，在容器內執行評估：
```bash
python evaluate_with_semantic_similarity.py
```
評估報告將會顯示在終端機，並生成一份詳細的 JSON 結果檔案。

## 🤝 貢獻指南

我們歡迎任何形式的貢獻！

### 如何貢獻
1. Fork 本專案
2. 建立功能分支: `git checkout -b feature/amazing-feature`
3. 提交變更: `git commit -m 'Add amazing feature'`
4. 推送分支: `git push origin feature/amazing-feature`
5. 開啟 Pull Request

## 📄 授權條款

本專案採用 MIT 授權條款 - 查看 `LICENSE` 檔案了解詳情。

## 📞 聯絡我們

- **問題回報**: [GitHub Issues](https://github.com/your-username/breeze-finetune-pipeline/issues)
