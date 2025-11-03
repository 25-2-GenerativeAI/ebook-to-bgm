"""
전처리 + LoRA 파인튜닝한 Stable Audio 를 만들 건데,

우선 1. model4_lora_prompts_short(이미 Model3대로 전처리, LLM프롬프트로 나온 것)를 Model3_generate2.py(기본 LoRA 없는 Baseline Stable Audio)에 넣어서 만들고 model4_lora_audio 폴더에 저장
2. 그 오디오를 model4_lora_prompts.csv 파일에 있는 텍스트만 골라서
3. 왜 적절한지/부적절한지 이유와 5점 척도를 rating해서 (1명만)
4. 그걸 lora 모델로, Model4 이름으로 빌드하고
5. Model4에서 prompts.json에 있는 3번째 프롬프트로 만들고 lora_outputs 폴더에 저장
6. 그걸 Model3에서 만든 3번째 오디오랑 비교하기(LoRA 하기 전/후를 비교해서 LoRA가 의미있었는지 확인하기 위함)
7. 추가로 그 부분에 해당하는 baseline도 Model5로 만들어서 3,4,5 비교하기

Model5: 슬라이딩 윈도우로 전처리하되 감정 같은 거 넣지 않고 자르기만 함 / LLM X / Stable Audio 기본, LoRA X
Model3: 슬라이딩 윈도우로 전처리 O / LLM O / Stable Audio 기본, LoRA X
Model4: 슬라이딩 윈도우로 전처리 O / LLM X / Stable Audio 기본, LoRA O (이때 LoRA는 Model3에서 생성된 프롬프트를 보고 인간이 rating한 것을 참고, 즉 LLM은 사용하긴 하지만 모델이 쓰는 건 프롬프트가 아니라 인간이 rating한 것.)
"""

"""
텍스트
SAMPSON. True, and therefore women, being the weaker vessels, are ever thrust to the wall: therefore I will push Montague’s men from the wall, and thrust his maids to the wall. 
GREGORY. The quarrel is between our masters and us their men. 
SAMPSON. ’Tis all one, I will show myself a tyrant: when I have fought with the men I will be civil with the maids, I will cut off their heads.

SAMPSON. 그렇지, 그러니까 여자들은 연약한 그릇이니 항상 벽 쪽으로 밀려나게 마련이야. 그러니 내가 몬태규 편 사람들을 벽에서 밀어내고, 그의 처녀들은 벽 쪽으로 떠밀어 놓겠어.
GREGORY. 싸움은 우리 주인들 사이의 일이요, 우리 종들은 그들 편이오.
SAMPSON. 다 똑같은 말이지, 내가 폭군처럼 굴어주지: 먼저 남자들과 싸우고 난 뒤에는 처녀들에게는 예의를 베풀겠다, 그리고 그들의 목을 벨 거야.

"Gentle solo piano with a bittersweet, reflective melody.", 1, "텍스트 자체가 폭력적인데 오디오가 너무 평화롭고 부드러움",
"Soft strings with rising tension, expressing wounded pride.", 2, "폭력적이고 긴박한 텍스트에 맞지 않게 오디오가 몽환적이고 신비로움", 
"Minimalist harp and flute, evoking irony and hurt feelings.", 3, "폭력적이고 긴박한 텍스트에 맞지 않게 신비롭고 조용함",
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from diffusers import StableAudioPipeline
from peft import LoraConfig, get_peft_model
import librosa

# -------------------------------
# Dataset
# -------------------------------
class AudioPromptDataset(Dataset):
    def __init__(self, json_path, tokenizer, sample_rate=48000):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt, audio_path, rating = item["prompt"], item["audio_path"], item["rating"]

        # 텍스트 토큰화
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77
        )

        # 오디오 로드 (waveform)
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        audio_tensor = torch.tensor(audio, dtype=torch.float32)

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "audio": audio_tensor,
            "rating": torch.tensor(rating, dtype=torch.float32),
        }

# -------------------------------
# Training
# -------------------------------
def train_lora(json_path="model4_lora_prompts_short.json", save_dir="model4_lora_weights", epochs=3, batch_size=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Stable Audio 로드
    pipe = StableAudioPipeline.from_pretrained(
        "stabilityai/stable-audio-open-1.0",
        torch_dtype=torch.float16,
    ).to(device)

    # LoRA 구성
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["to_q", "to_v"],
    )
    pipe.unet = get_peft_model(pipe.unet, lora_config)

    # Dataset / DataLoader
    dataset = AudioPromptDataset(json_path, pipe.tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer & Loss
    optimizer = optim.AdamW(pipe.unet.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()

    pipe.unet.train()

    for epoch in range(epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            audio = batch["audio"].to(device)  # (B, T)
            rating = batch["rating"].to(device)

            # 텍스트 임베딩
            text_embeds = pipe.text_encoder(input_ids, attention_mask=attention_mask)[0]

            # 오디오 latent (VAE encode)
            audio = audio.unsqueeze(1)  # (B,1,T)
            audio_latent = pipe.vae.encode(audio.half()).latent_dist.sample()

            # Noise
            noise = torch.randn_like(audio_latent)
            timesteps = torch.randint(0, pipe.scheduler.num_train_timesteps, (1,), device=device).long()
            noisy_latents = pipe.scheduler.add_noise(audio_latent, noise, timesteps)

            # UNet 예측
            noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeds).sample

            # 손실 (rating 가중치 적용)
            loss = mse_loss(noise_pred, noise) * (rating / 5.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    # LoRA 가중치 저장
    os.makedirs(save_dir, exist_ok=True)
    pipe.unet.save_pretrained(save_dir)
    print(f"✅ LoRA weights saved to {save_dir}")

# -------------------------------
# 실행부
# -------------------------------
if __name__ == "__main__":
    train_lora(
        json_path="model4_lora_prompts_short.json",
        save_dir="model4_lora_weights",
        epochs=3,
        batch_size=1
    )
