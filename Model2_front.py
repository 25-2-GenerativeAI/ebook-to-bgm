import os
from dotenv import load_dotenv
import pandas as pd
from collections import defaultdict
from transformers import pipeline
from openai import OpenAI
import torch
from diffusers import AudioLDM2Pipeline
import soundfile as sf
import torch.nn as nn

# .env 파일 불러오기
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print("API KEY:", api_key)

# -------------------------------
# Step 1. 텍스트 처리 함수
# -------------------------------

def load_text(file_path: str) -> str:
    """텍스트 파일 불러오기"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def split_into_paragraphs(text: str) -> list[str]:
    """빈 줄(\\n\\n)을 기준으로 문단 분리"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs

def sliding_window(paragraphs: list[str], window_size: int = 3, stride: int = 1) -> list[list[str]]:
    """슬라이딩 윈도우 생성"""
    windows = []
    for i in range(0, len(paragraphs) - window_size + 1, stride):
        windows.append(paragraphs[i:i+window_size])
    return windows

# -------------------------------
# Step 2. 감정 분류기 (RoBERTa)
# -------------------------------

emotion_model_name = "j-hartmann/emotion-english-distilroberta-base"

emotion_classifier = pipeline(
    "text-classification",
    model=emotion_model_name,
    top_k=1,
)

def detect_emotion(text: str) -> str:
    """BERT 기반 감정 분류"""
    result = emotion_classifier(text[:512])  
    if isinstance(result[0], list):  
        return result[0][0]["label"]
    else:
        return result[0]["label"]

# -------------------------------
# Step 3. LSN 데이터 불러오기
# -------------------------------

def load_sensorimotor_db(path: str) -> dict:
    """
    Lancaster Sensorimotor Norms 불러오기
    CSV/XLSX 모두 지원
    반환: {단어: {감각: 점수}}
    """
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    # 컬럼 이름 매핑 (LSN → 우리 코드)
    sense_map = {
        "Auditory.mean": "hearing",
        "Gustatory.mean": "taste",
        "Haptic.mean": "touch",
        "Interoceptive.mean": "interoception",
        "Olfactory.mean": "smell",
        "Visual.mean": "vision",
    }

    db = {}
    for _, row in df.iterrows():
        word = str(row["Word"]).lower()
        db[word] = {
            sense_map[col]: float(row[col])
            for col in sense_map if col in row and not pd.isna(row[col])
        }
    return db

# -------------------------------
# Step 4. 감각 태깅 (LSN 기반)
# -------------------------------

def detect_senses(text: str, db: dict, threshold: float = 3.0) -> list[str]:
    """
    LSN 기반 sensory 태깅
    threshold 이상 평균값인 감각을 반환
    """
    tokens = text.lower().split()  # 공백 기준 단순 토큰화
    scores = defaultdict(float)
    counts = defaultdict(int)

    for tok in tokens:
        if tok in db:
            for sense, score in db[tok].items():
                scores[sense] += score
                counts[sense] += 1

    senses = []
    for sense, total_score in scores.items():
        avg_score = total_score / counts[sense]
        if avg_score >= threshold:
            senses.append(sense)
    return senses

# -------------------------------
# Step 5. 윈도우 태깅
# -------------------------------

import re
from collections import Counter

def tag_window(window: list[str], db: dict) -> dict:
    """윈도우(여러 문단)에 대해 감정+감각 태깅 (개선 버전)"""
    combined_text = " ".join(window)

    # 문장 단위로 분리
    sentences = re.split(r"[.!?]\s+", combined_text)
    emotions = []
    for sent in sentences:
        sent = sent.strip()
        if sent:
            emotions.append(detect_emotion(sent))

    # 다수결 (문장이 없으면 neutral)
    if emotions:
        emotion = Counter(emotions).most_common(1)[0][0]
    else:
        emotion = "neutral"

    # 감각 태깅 (구두점 제거 + threshold 낮춤)
    cleaned_text = re.sub(r"[^a-zA-Z\s]", "", combined_text.lower())
    senses = detect_senses(cleaned_text, db, threshold=2.5)

    return {
        "text": combined_text,
        "emotion": emotion,
        "senses": senses
    }


import json

if __name__ == "__main__":
    # 1. 텍스트 → 문단/윈도우
    text = load_text("data.txt")
    paragraphs = split_into_paragraphs(text)
    windows = sliding_window(paragraphs, window_size=3, stride=1)

    # 2. LSN DB 로드
    lsn_db_path = "LSN.csv"   # 👉 실제 경로 맞게 수정
    lsn_db = load_sensorimotor_db(lsn_db_path)

    # 3. 각 윈도우 태깅
    tagged = []
    for i, window in enumerate(windows):
        tags = tag_window(window, lsn_db)
        tagged.append(tags)

    # 4. JSON으로 저장
    with open("windows.json", "w", encoding="utf-8") as f:
        json.dump(tagged, f, ensure_ascii=False, indent=2)

    print(f"✅ {len(tagged)} windows saved to windows.json")
