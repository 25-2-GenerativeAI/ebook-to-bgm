from diffusers import AudioLDMPipeline
import torch
import scipy.io.wavfile

def generate_audio(prompt: str, output_path="output.wav", duration=10, steps=50):
    # 모델 불러오기
    pipe = AudioLDMPipeline.from_pretrained(
        "cvssp/audioldm2-large", 
        torch_dtype=torch.float16
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # 음원 생성
    audio = pipe(
        prompt,
        num_inference_steps=steps,
        audio_length_in_s=duration
    ).audios[0]

    # 샘플링 레이트 (기본값 16000Hz)
    sampling_rate = pipe.vae.config["sampling_rate"]

    # wav 파일로 저장
    scipy.io.wavfile.write(output_path, sampling_rate, audio)

    print(f"✅ Audio generated and saved at {output_path}")


if __name__ == "__main__":
    test_prompt = "A calm piano melody with soft rain in the background"
    generate_audio(test_prompt, "test_output.wav", duration=10)
