import os
import json
import torch
import soundfile as sf
import numpy as np
import librosa
from transformers import ClapProcessor, ClapModel

def evaluate_clap(audio_dir="outputs/audio", prompt_json="prompts.json", out_json="clap_scores.json"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CLAP 모델 로드
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)

    # 프롬프트 불러오기
    with open(prompt_json, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    results = []
    scores = []

    for idx, prompt in enumerate(prompts, start=1):
        audio_path = os.path.join(audio_dir, f"scene_{idx:03}.wav")
        if not os.path.exists(audio_path):
            print(f"⚠️ Audio file not found: {audio_path}")
            continue

        # 오디오 로드 (원래는 16kHz)
        audio, sr = sf.read(audio_path)

        # 리샘플링 (16kHz → 48kHz)
        if sr != 48000:
            audio = librosa.resample(audio.astype(float), orig_sr=sr, target_sr=48000)
            sr = 48000

        # CLAP 입력 준비
        inputs = processor(text=[prompt], audios=[audio], return_tensors="pt", padding=True, sampling_rate=sr)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 임베딩 추출
        with torch.no_grad():
            outputs = model(**inputs)
            audio_embeds = outputs.audio_embeds
            text_embeds = outputs.text_embeds

        # 코사인 유사도
        sim = torch.nn.functional.cosine_similarity(audio_embeds, text_embeds).item()
        results.append({"scene_id": idx, "prompt": prompt, "clap_score": sim})
        scores.append(sim)

        print(f"🎵 Scene {idx}: CLAP score = {sim:.4f}")

    # 📊 전체 지표 계산
    if scores:
        avg_score = float(np.mean(scores))
        max_score = float(np.max(scores))
        variance = float(np.var(scores))
    else:
        avg_score, max_score, variance = 0.0, 0.0, 0.0

    print(f"\n📊 Average CLAP score = {avg_score:.4f}")
    print(f"🏆 Max CLAP score = {max_score:.4f}")
    print(f"σ² Variance = {variance:.6f}")

    # JSON 저장
    output = {
        "results": results,
        "average_score": avg_score,
        "max_score": max_score,
        "variance": variance
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ Scores saved to {out_json}")
    return output

if __name__ == "__main__":
    evaluate_clap()
