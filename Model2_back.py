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
# Step 6. 프롬프트 생성
# -------------------------------

client = OpenAI()

def generate_prompt(emotion: str, senses: list[str], text: str) -> str:
    """
    GPT-5-nano를 이용해 최종 오디오 생성 프롬프트 생성
    """
    sense_str = ", ".join(senses) if senses else "none"
    system_msg = """You are a music prompt generator.
Given an emotion, sensory tags, and a text excerpt,
analyze the mood and narrative structure,
then output a short, vivid English prompt suitable for an audio diffusion model."""
    
    user_msg = f"""
Emotion: {emotion}
Senses: {sense_str}
Text excerpt: {text[:200]}...
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.7,
        max_completion_tokens=100,
    )
    if response.choices and response.choices[0].message:
        return response.choices[0].message.content or "(empty)"
    else:
        return "(no response)"

# -------------------------------
# Step 7. 오디오 생성 (AudioLDM2)
# -------------------------------

def generate_audio_from_prompt(prompt: str, output_path="output.wav", duration=10, steps=50):
    """AudioLDM2로 오디오 생성"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2-large",
        torch_dtype=torch.float16,
    ).to(device)

    audio = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        audio_length_in_s=duration,
        guidance_scale=2.0,
        generator=torch.manual_seed(0),
    ).audios[0]

    # AudioLDM2 기본 샘플링 레이트는 16kHz
    sampling_rate = 16000
    sf.write(output_path, audio, sampling_rate)
    print(f"✅ Saved audio at {output_path}")

