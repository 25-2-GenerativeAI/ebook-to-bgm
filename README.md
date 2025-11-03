## ebook-to-bgm
**Ebook 감정 분석 기반 BGM 자동 생성 프로젝트**

## 📌 모델 & 기술 흐름 요약
1. **텍스트 입력 (Ebook 페이지 단위)**
    - 소설 텍스트를 입력받음
    - 직접 쓰기 vs 감정/분위기 태깅 후 요약된 prompt 사용 → 두 가지 방법을 비교 실험
    - (예: "주인공이 배신당하고 절망하는 장면" → "dark, suspenseful, dramatic")
2. **텍스트 인코딩**
    - *사전학습된 언어모델 (BERT, DistilBERT, RoBERTa 등)**으로 감정/분위기 임베딩 추출
    - 추가로 감정 분류기/zero-shot classifier를 붙여 **mood label 생성** 가능
3. **음악 생성 모델**
    - **Meta의 MusicGen** (Hugging Face 공개됨, 사전학습 모델 사용)
        - 입력: 텍스트 prompt (직접 텍스트 또는 추출된 mood 태그)
        - 출력: 오디오 wave 형태의 음악
    - **LoRA / Adapter / Test-time prompt tuning**으로 **low-resource 조건**에서 성능 개선 시도
4. **Low-resource Adaptation 전략**
    - Training-free baseline: 그냥 MusicGen + 텍스트 입력
    - Adapter/LoRA 방식: 소설 텍스트에 특화된 "문체별 감정-음악 매핑"을 소규모 데이터로 추가 학습
    - Test-time adaptation: 새로운 텍스트가 들어오면 즉석에서 prompt를 자동 보정 (예: 감정 태그 강화)
5. **출력**
    - Ebook 페이지마다 대응되는 짧은 BGM 오디오
    - 결과 비교:
        - (a) Raw 텍스트 → MusicGen
        - (b) 감정 태깅 + 요약 프롬프트 → MusicGen
