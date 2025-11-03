import os
import json
import torch
import soundfile as sf
from diffusers import StableAudioPipeline
import numpy as np

# -------------------------------
# 오디오 생성 함수
# -------------------------------
def generate_audio_from_prompt(pipe, prompt: str, output_path="output.wav", duration=10, steps=50):
    # 시드 고정 (재현성)
    generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(0)

    # 오디오 생성
    audio = pipe(
        prompt=prompt,
        negative_prompt="low quality, noisy, distorted",
        num_inference_steps=steps,
        audio_end_in_s=duration,
        generator=generator,
    ).audios[0]

    # numpy 변환
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = audio.astype(np.float32)

    # Stable Audio는 48kHz 기본
    sr = pipe.vae.sampling_rate
    sf.write(output_path, audio.T, sr)
    print(f"✅ Saved audio at {output_path} ({sr}Hz)")

# -------------------------------
# 실행부
# -------------------------------
def main(input_json="model4_lora_prompts_short.json", out_dir="model3_lora_audio"):
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"{input_json} not found.")

    with open(input_json, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Stable Audio 파이프라인 로드
    pipe = StableAudioPipeline.from_pretrained(
        "stabilityai/stable-audio-open-1.0",
        torch_dtype=torch.float16,
    ).to(device)

    # 오디오 저장 폴더
    audio_dir = os.path.join(out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    # prompts.json이 문자열 리스트라고 가정
    for idx, prompt in enumerate(prompts, start=1):
        print(f"\n🎧 Generating audio for Scene {idx}...")
        output_file = os.path.join(audio_dir, f"scene_{idx:03}.wav")
        generate_audio_from_prompt(pipe, prompt, output_path=output_file, duration=10, steps=50)

    print(f"\n✅ All audios generated successfully! Saved in '{audio_dir}'")

if __name__ == "__main__":
    main()
