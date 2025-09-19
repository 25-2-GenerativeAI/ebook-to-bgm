import os
import soundfile as sf
import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, ClapProcessor, ClapModel

# -------------------------
# 1. 데이터셋 로드 (data.txt 파일 단일 사용)
# -------------------------
def load_single_text(file_path: str, encoding="utf-8"):
    with open(file_path, "r", encoding=encoding) as f:
        text = f.read().strip()
    return {"file": os.path.basename(file_path), "text": text}


# -------------------------
# 2. 전처리: 문단 분리 + 슬라이딩 윈도우
# -------------------------
def split_into_paragraphs(text: str):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs

def sliding_window(paragraphs, window_size=3, stride=1):
    windows = []
    for i in range(0, len(paragraphs) - window_size + 1, stride):
        window = paragraphs[i:i+window_size]
        windows.append(window)
    return windows


# -------------------------
# 3. 감정 태깅 (Sensory T5)
# -------------------------
emotion_model = "h2oai/sensory-t5-small"
tokenizer_emotion = AutoTokenizer.from_pretrained(emotion_model)
model_emotion = AutoModelForSeq2SeqLM.from_pretrained(emotion_model)

def detect_emotion(text):
    prompt = f"Detect emotion in the following text:\n\n{text}"
    inputs = tokenizer_emotion(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model_emotion.generate(**inputs, max_new_tokens=32)
    emotion = tokenizer_emotion.decode(outputs[0], skip_special_tokens=True)
    return emotion


# -------------------------
# 4. GPT-5-nano: 분위기 + 서사구조 + 최종 프롬프트 생성
# -------------------------
from dotenv import load_dotenv
import os

# .env 파일 불러오기
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print("API KEY:", api_key)

client = OpenAI()

def generate_prompt_with_gpt(text, emotion):
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": "You are an assistant that creates audio narration prompts from novel text."},
            {"role": "user", "content": f"""
Given the following text and its detected emotion:

Emotion: {emotion}
Text: {text}

1. Detect the mood
2. Detect the narrative structure
3. Create a final narration prompt that integrates text, emotion, mood, and narrative style.
Only return the final narration prompt, nothing else.
"""}
        ],
        max_tokens=300
    )
    return response.choices[0].message.content.strip()


# -------------------------
# 5. 오디오 생성 (AudioLDM2)
# -------------------------
pipe = pipeline("text-to-audio", model="cvssp/audioldm2")

def generate_audio_from_text(text, window_size=3, stride=1, out_dir="./outputs"):
    os.makedirs(out_dir, exist_ok=True)

    paragraphs = split_into_paragraphs(text)
    windows = sliding_window(paragraphs, window_size, stride)

    audio_files = []
    for idx, w in enumerate(windows):
        w_text = " ".join(w)

        # 1) 감정 태깅
        emotion = detect_emotion(w_text)

        # 2) GPT-5-nano로 프롬프트 생성
        prompt = generate_prompt_with_gpt(w_text, emotion)
        print(f"\n[Window {idx}] Generated Prompt:\n{prompt}")

        # 3) 오디오 생성
        audio = pipe(prompt, forward_params={"max_new_tokens": 1024})
        file_path = os.path.join(out_dir, f"output_{idx}.wav")
        sf.write(file_path, audio["audio"], audio["sampling_rate"])
        audio_files.append((file_path, prompt))
    return audio_files


# -------------------------
# 6. 성능 평가: CLAP Score
# -------------------------
clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

def evaluate_clap(audio_path, prompt):
    audio_input, sr = sf.read(audio_path)  # wav 읽기
    inputs = clap_processor(text=[prompt], audios=[audio_input], sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clap_model(**inputs)
        # text와 audio 임베딩 사이 코사인 유사도
        text_embeds = outputs.text_embeds
        audio_embeds = outputs.audio_embeds
        score = torch.nn.functional.cosine_similarity(text_embeds, audio_embeds).item()
    return score


# -------------------------
# 7. 실행
# -------------------------
if __name__ == "__main__":
    # data.txt 파일 불러오기
    dataset = load_single_text("data.txt")

    # 오디오 생성
    audio_files = generate_audio_from_text(dataset["text"], window_size=3, stride=1)

    # CLAP Score 평가
    for file_path, prompt in audio_files:
        score = evaluate_clap(file_path, prompt)
        print(f"[CLAP] {file_path} → Score: {score:.4f}")
