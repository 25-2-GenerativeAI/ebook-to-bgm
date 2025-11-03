import os
import json
import torch
import soundfile as sf
from diffusers import AudioLDMPipeline  # ✅ v1 파이프라인
import numpy as np

# -------------------------------
# 오디오 생성 함수
# -------------------------------
def generate_audio_from_prompt(pipe, prompt: str, output_path="output.wav", duration=10, steps=50):
    audio = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        audio_length_in_s=duration,
        guidance_scale=2.0,
        generator=torch.manual_seed(0),
    ).audios[0]

    # AudioLDM v1 기본 샘플링 레이트는 16kHz
    sr = pipe.vae.config.get("sampling_rate", 16000)

    # numpy 변환 보정
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = audio.astype(np.float32)

    sf.write(output_path, audio, sr)
    print(f"✅ Saved audio at {output_path} ({sr}Hz)")

# -------------------------------
# 실행부
# -------------------------------
def main(input_json="prompts.json", out_dir="outputs"):
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"{input_json} not found.")

    with open(input_json, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    # ✅ 파이프라인 한 번만 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = AudioLDMPipeline.from_pretrained(
        "cvssp/audioldm-large",   # ⚡ v1 모델 (안정적)
        torch_dtype=torch.float16,
    ).to(device)

    # ✅ 오디오 저장 폴더 생성
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
