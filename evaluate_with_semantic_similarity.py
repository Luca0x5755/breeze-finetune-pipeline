# ==============================================================================
# 檔案：evaluate_with_semantic_similarity.py
# 描述：使用向量相似度評估語音辨識模型的語義表達準確度
# ==============================================================================
import json
import os
from datetime import datetime

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Audio
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class SemanticSimilarityEvaluator:
    """語義相似度評估器"""

    def __init__(self, embedding_model_name="shibing624/text2vec-base-chinese"):
        """
        初始化評估器

        Args:
            embedding_model_name (str): 用於生成向量的模型名稱
                推薦選項：
                - "shibing624/text2vec-base-chinese" (中文專用)
                - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (多語言)
                - "DMetaSoul/sbert-chinese-qmc-domain-v1" (中文對話)
        """
        print(f"正在載入向量模型：{embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.results = []

    def compute_similarity(self, text1, text2):
        """
        計算兩個文本的語義相似度

        Args:
            text1 (str): 參考文本
            text2 (str): 預測文本

        Returns:
            float: 相似度分數 (0-1之間，1表示完全相似)
        """
        if not text1.strip() or not text2.strip():
            return 0.0

        # 生成向量
        embeddings = self.embedding_model.encode([text1, text2])

        # 計算餘弦相似度
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)


def evaluate_with_semantic_similarity(
    model_path: str,
    test_folder_path: str,
    reference_csv_path: str,
    embedding_model: str = "shibing624/text2vec-base-chinese",
    output_results: bool = True,
    language: str = "zh",
):
    """
    使用向量相似度評估語音辨識模型

    Args:
        model_path (str): 訓練好的模型路徑
        test_folder_path (str): 測試音訊檔案資料夾
        reference_csv_path (str): 參考文本CSV檔案
        embedding_model (str): 向量模型名稱
        output_results (bool): 是否輸出詳細結果
        language (str): 目標語言
    """

    # --- 1. 初始化模型和環境 ---
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"將使用設備：{device}")

    print(f"\n--- 正在載入語音辨識模型：{model_path} ---")
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)

    # 初始化語義相似度評估器
    similarity_evaluator = SemanticSimilarityEvaluator(embedding_model)

    # --- 2. 準備參考答案 ---
    print(f"\n--- 正在讀取參考文本：{reference_csv_path} ---")
    reference_df = pd.read_csv(reference_csv_path)
    reference_dict = pd.Series(
        reference_df.transcription.values,
        index=reference_df.file.apply(lambda x: os.path.basename(x)),
    ).to_dict()

    # --- 3. 收集測試檔案並進行評估 ---
    predictions = []
    references = []
    similarity_scores = []
    detailed_results = []

    # 找出所有 .wav 檔案
    test_files = [f for f in os.listdir(test_folder_path) if f.endswith(".wav")]
    if not test_files:
        print(f"錯誤：在資料夾 {test_folder_path} 中找不到任何 .wav 檔案。")
        return

    print(f"\n--- 找到 {len(test_files)} 個 .wav 檔案，開始進行評估 ---")

    audio_loader = Audio(sampling_rate=16000)

    # 使用進度條進行批量處理
    for filename in tqdm(test_files, desc="語義相似度評估進度"):
        if filename not in reference_dict:
            continue

        file_path = os.path.join(test_folder_path, filename)
        reference_text = reference_dict[filename]

        try:
            # 載入並處理音訊
            audio_input = audio_loader.decode_example(
                audio_loader.encode_example(file_path)
            )
            input_features = processor(
                audio_input["array"],
                sampling_rate=audio_input["sampling_rate"],
                return_tensors="pt",
            ).input_features.to(device)

            # 進行語音辨識
            predicted_ids = model.generate(input_features)
            predicted_text = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

            # 計算語義相似度
            similarity_score = similarity_evaluator.compute_similarity(
                reference_text, predicted_text
            )

            # 收集結果
            predictions.append(predicted_text)
            references.append(reference_text)
            similarity_scores.append(similarity_score)

            # 詳細結果記錄
            detailed_results.append(
                {
                    "filename": filename,
                    "reference": reference_text,
                    "prediction": predicted_text,
                    "similarity_score": similarity_score,
                }
            )

        except Exception as e:
            print(f"處理檔案 {filename} 時發生錯誤: {e}")

    # --- 4. 計算並顯示評估結果 ---
    if not predictions:
        print("\n錯誤：沒有任何檔案被成功處理。")
        return

    print("\n--- 正在計算評估指標 ---")

    # 傳統 WER 計算
    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(predictions=predictions, references=references) * 100

    # 語義相似度統計
    mean_similarity = np.mean(similarity_scores)
    std_similarity = np.std(similarity_scores)
    high_similarity_count = sum(1 for score in similarity_scores if score >= 0.8)
    moderate_similarity_count = sum(
        1 for score in similarity_scores if 0.6 <= score < 0.8
    )
    low_similarity_count = sum(1 for score in similarity_scores if score < 0.6)

    # --- 5. 顯示結果 ---
    print("\n" + "=" * 60)
    print("                   評估結果總結")
    print("=" * 60)
    print(f"總共評估檔案數量: {len(predictions)}")
    print(f"使用向量模型: {embedding_model}")
    print()
    print("🎯 傳統指標:")
    print(f"   字詞錯誤率 (WER): {wer:.2f}%")
    print()
    print("🧠 語義相似度指標:")
    print(f"   平均相似度: {mean_similarity:.4f}")
    print(f"   標準差: {std_similarity:.4f}")
    print(
        f"   高相似度 (≥0.8): {high_similarity_count} 個 ({high_similarity_count/len(predictions)*100:.1f}%)"
    )
    print(
        f"   中等相似度 (0.6-0.8): {moderate_similarity_count} 個 ({moderate_similarity_count/len(predictions)*100:.1f}%)"
    )
    print(
        f"   低相似度 (<0.6): {low_similarity_count} 個 ({low_similarity_count/len(predictions)*100:.1f}%)"
    )

    # 相似度分數分布
    print()
    print("📊 相似度分數分布:")
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for i in range(len(bins) - 1):
        count = sum(1 for score in similarity_scores if bins[i] <= score < bins[i + 1])
        percentage = count / len(predictions) * 100
        print(f"   {bins[i]:.1f}-{bins[i+1]:.1f}: {count} 個 ({percentage:.1f}%)")

    print("=" * 60)

    # --- 6. 輸出詳細結果 ---
    if output_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"semantic_similarity_results_{timestamp}.json"

        results_summary = {
            "evaluation_time": timestamp,
            "model_path": model_path,
            "embedding_model": embedding_model,
            "total_files": len(predictions),
            "wer": wer,
            "mean_similarity": mean_similarity,
            "std_similarity": std_similarity,
            "high_similarity_count": high_similarity_count,
            "moderate_similarity_count": moderate_similarity_count,
            "low_similarity_count": low_similarity_count,
            "detailed_results": detailed_results,
        }

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results_summary, f, ensure_ascii=False, indent=2)

        print(f"\n📄 詳細結果已儲存至: {results_file}")

        # 顯示最佳和最差的幾個例子
        sorted_results = sorted(
            detailed_results, key=lambda x: x["similarity_score"], reverse=True
        )

        print("\n🏆 相似度最高的 3 個例子:")
        for i, result in enumerate(sorted_results[:3], 1):
            print(f"  {i}. 相似度: {result['similarity_score']:.4f}")
            print(f"     參考: {result['reference']}")
            print(f"     預測: {result['prediction']}")
            print()

        print("⚠️  相似度最低的 3 個例子:")
        for i, result in enumerate(sorted_results[-3:], 1):
            print(f"  {i}. 相似度: {result['similarity_score']:.4f}")
            print(f"     參考: {result['reference']}")
            print(f"     預測: {result['prediction']}")
            print()

    return {
        "wer": wer,
        "mean_similarity": mean_similarity,
        "similarity_scores": similarity_scores,
        "detailed_results": detailed_results,
    }


if __name__ == "__main__":
    # --- 請修改以下參數 ---

    # 訓練好的模型路徑
    MY_FINE_TUNED_MODEL_PATH = "./whisper-small-zh-finetune-final"

    # 測試音訊檔案資料夾
    TEST_FOLDER_PATH = "path/to/your/test_wavs"

    # 參考文本CSV檔案
    REFERENCE_CSV = "output/final_audio_paths.csv"

    # 向量模型選擇 (可以嘗試不同的模型)
    EMBEDDING_MODELS = [
        "shibing624/text2vec-base-chinese",  # 中文專用，效果佳
        # "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 多語言支援
        # "DMetaSoul/sbert-chinese-qmc-domain-v1",    # 中文對話領域
    ]

    # 對每個向量模型進行評估
    for embedding_model in EMBEDDING_MODELS:
        print(f"\n{'='*80}")
        print(f"使用向量模型: {embedding_model}")
        print(f"{'='*80}")

        results = evaluate_with_semantic_similarity(
            model_path=MY_FINE_TUNED_MODEL_PATH,
            test_folder_path=TEST_FOLDER_PATH,
            reference_csv_path=REFERENCE_CSV,
            embedding_model=embedding_model,
            output_results=True,
        )
