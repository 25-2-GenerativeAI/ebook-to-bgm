from diffusers import AudioLDM2Pipeline
import torch
import scipy

# 모델 로드
pipe = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2", torch_dtype=torch.float16)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# 예시 프롬프트
prompt = "A mysterious and dark ambient background music with deep atmospheric tones"

# 오디오 생성 (10초 길이)
audio = pipe(
    prompt, 
    num_inference_steps=50, 
    audio_length_in_s=10, 
    generate_prompt=False
).audios[0]


# wav 파일 저장
scipy.io.wavfile.write("output.wav", rate=16000, data=audio)
