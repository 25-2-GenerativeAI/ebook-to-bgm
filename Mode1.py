"""
Romeo & Juliet 기반 Ebook BGM 생성 파이프라인
1. 데이터 전처리 (슬라이딩 윈도우 + 감정/분위기/서사 구조)
2. 프롬프트 생성 (AI 기반)
3. MusicGen 음악 생성
4. Best-of-N / RLAIF 선택
5. CLAP Score 평가
"""

from transformers import pipeline
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from openai import OpenAI
import torch
import scipy.io.wavfile
import numpy as np
import random
import soundfile as sf

# ========== 1. 데이터 전처리 ==========
# Romeo & Juliet 예시 텍스트 (일부)
text = """
But soft, what light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief,
That thou her maid art far more fair than she.
"""

# (1) 텍스트 분리 → 슬라이딩 윈도우
def sliding_window_split(text, window_size=3):
    sentences = text.split("\n")
    sentences = [s.strip() for s in sentences if s.strip()]
    windows = [ " ".join(sentences[i:i+window_size]) 
                for i in range(0, len(sentences), window_size) ]
    return windows

segments = sliding_window_split(text, window_size=2)
print(segments[0])
print(segments[1])
print(segments[2])

# (2) 감정 + 분위기 + 서사 구조 라벨링
emotion_classifier = pipeline("text-classification",
                              model="cardiffnlp/twitter-roberta-base-emotion",
                              return_all_scores=False)

def label_segments(segments):
    labeled = []
    for seg in segments:
        emotion = emotion_classifier(seg)[0]['label']
        # 분위기: 단순 rule (추후 모델 대체 가능)
        if "dark" in seg.lower() or "grief" in seg.lower():
            tone = "suspenseful"
        else:
            tone = "romantic"
        # 서사 구조: LLM 사용 (여기선 임시 rule)
        if "Arise" in seg:
            stage = "rising action"
        else:
            stage = "introduction"
        labeled.append({"text": seg, "emotion": emotion, "tone": tone, "stage": stage})
    return labeled

labeled_segments = label_segments(segments)
print(labeled_segments[0])
print(labeled_segments[1])
print(labeled_segments[2])

# ========== 2. 프롬프트 생성 (AI 기반) ==========
# 규칙 기반 + LLM 기반 혼합 (OpenAI API 예시)
from dotenv import load_dotenv
import os

# .env 파일 불러오기
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print("API KEY:", api_key)

client = OpenAI()

def generate_prompt_ai(emotion, tone, stage):
    system_prompt = "You are a music prompt generator for MusicGen."
    user_prompt = f"Emotion: {emotion}, Tone: {tone}, Narrative Stage: {stage}. \
Create a music prompt for background soundtrack."
    resp = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role":"system","content":system_prompt},
                  {"role":"user","content":user_prompt}]
    )
    return resp.choices[0].message.content

for seg in labeled_segments:
    seg["prompt"] = generate_prompt_ai(seg["emotion"], seg["tone"], seg["stage"])

# ========== 3. MusicGen 생성 ==========
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
musicgen_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

def generate_music(prompt, idx, duration=10):
    inputs = processor(text=[prompt], return_tensors="pt")
    audio_values = musicgen_model.generate(**inputs, max_new_tokens=512)  # float32 [-1, 1]

    # numpy 변환
    audio = audio_values[0, 0].cpu().numpy()

    # wav 저장
    sf.write(
        f"music_{idx}.wav",
        audio,
        samplerate=musicgen_model.config.audio_encoder.sampling_rate,
        subtype="PCM_16"
    )
    print(f"✅ music_{idx}.wav 저장 완료")

# 실행
for i, seg in enumerate(labeled_segments):
    generate_music(seg["prompt"], i)

# ========== 4. Best-of-N & RLAIF ==========
# Best-of-N: N개 후보 생성 후 점수 가장 높은 것 선택
def best_of_n(prompt, n=3):
    candidates = []
    for i in range(n):
        inputs = processor(text=[prompt], return_tensors="pt")
        audio_values = musicgen_model.generate(**inputs, max_new_tokens=512)
        filename = f"candidate_{i}.wav"
        scipy.io.wavfile.write(filename,
                               rate=musicgen_model.config.audio_encoder.sampling_rate,
                               data=audio_values[0].cpu().numpy())
        candidates.append(filename)
    # RLAIF: CLAP 기반 점수로 rerank
    # (여기선 dummy: 랜덤 선택)
    return random.choice(candidates)

best_music = best_of_n(labeled_segments[0]["prompt"], n=3)

# ========== 5. CLAP Score 평가 ==========
# CLAP 모델 로드 (huggingface: laion/clap)
from transformers import ClapProcessor, ClapModel

clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

def evaluate_clap(prompt, audio_path):
    inputs = clap_processor(text=[prompt], audios=[audio_path], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clap_model(**inputs)
        score = torch.cosine_similarity(outputs.text_embeds, outputs.audio_embeds).item()
    return score

score = evaluate_clap(labeled_segments[0]["prompt"], best_music)
print("CLAP score:", score)

# ==== #
import soundfile as sf
import numpy as np
import librosa  # pip install librosa

def evaluate_clap(prompt, audio_path):
    # wav 파일 로드
    audio_waveform, sr = sf.read(audio_path)

    # stereo → mono 변환
    if audio_waveform.ndim > 1:
        audio_waveform = np.mean(audio_waveform, axis=1)

    # CLAP이 요구하는 48kHz로 리샘플링
    target_sr = 48000
    if sr != target_sr:
        audio_waveform = librosa.resample(audio_waveform, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # 텍스트만 따로 토큰화 (잘림 허용)
    text_inputs = clap_processor.tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    # 오디오는 feature extractor만
    audio_inputs = clap_processor.feature_extractor(
        [audio_waveform],
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )

    # 합치기
    inputs = {**text_inputs, **audio_inputs}

    with torch.no_grad():
        outputs = clap_model(**inputs)
        score = torch.cosine_similarity(outputs.text_embeds, outputs.audio_embeds).item()
    return score


def evaluate_clap_wobestofn(labeled_segments):
    results = []
    for i, seg in enumerate(labeled_segments):
        audio_path = f"music_{i}.wav"
        score = evaluate_clap(seg["prompt"], audio_path)
        results.append({"idx": i, "prompt": seg["prompt"], "score": score})
        print(f"[{i}] {audio_path} → CLAP score: {score:.4f}")
    return results

# 실행
scores = evaluate_clap_wobestofn(labeled_segments)
