from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
import scipy
import numpy as np
import soundfile as sf

# 디바이스 설정 (MPS 사용 가능하면 GPU 가속, 아니면 CPU)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Hugging Face에서 musicgen-small 모델 다운로드
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(device)

# ======= #

prompt = "romantic sad piano background music, slow tempo"

inputs = processor(
    text=[prompt],
    padding=True,
    return_tensors="pt"
).to(device)

# 길이를 늘려서 오디오 생성 (예: 8초)
audio_values = model.generate(**inputs, max_new_tokens=1024)

# ===== 5. 후처리 =====
audio = audio_values[0].cpu().numpy()

# (1, samples) → (samples,)
if audio.ndim > 1:
    audio = audio.squeeze(0)

# float32 변환
audio = audio.astype(np.float32)

# ===== 6. 저장 =====
sf.write("output.wav", audio, 32000, subtype="PCM_16")

print("✅ 음악 파일 생성 완료: output.wav")
