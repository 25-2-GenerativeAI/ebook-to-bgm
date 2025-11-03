import torch
from transformers import pipeline

# 디바이스 설정 (MPS 가능하면 MPS 사용, 아니면 CPU)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# 감정 분류 파이프라인 로드
classifier = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-emotion",
    device=0 if device == "mps" else -1
)

# Romeo & Juliet 샘플 텍스트
text_samples = [
    "But soft, what light through yonder window breaks?",
    "It is the east, and Juliet is the sun.",
    "Arise, fair sun, and kill the envious moon.",
    "Who is already sick and pale with grief.",
]

# 감정 분류 실행
for line in text_samples:
    result = classifier(line)
    print(f"Text: {line}")
    print("Prediction:", result, "\n")
