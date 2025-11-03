## ebook-to-bgm
**Ebook 감정 분석 기반 BGM 자동 생성 프로젝트**

## 📌 모델 & 기술 흐름 요약
1. **텍스트 입력**
    - 문단별로 감정 태깅 후, 이전 감정과 동일하면 붙이고 아니면 나누는 **슬라이딩 윈도우** 방식 사용
2. **텍스트 인코딩**
    - LLM 모델로 감정/분위기를 json 형태로 붙임
3. **음악 생성 모델**
    - Stable Audio 모델(diffusion 모델)
4. **LoRA**
    - Model5: 슬라이딩 윈도우로 전처리하되 감정 같은 거 넣지 않고 자르기만 함 / LLM X / Stable Audio 기본, LoRA X
    - Model3: 슬라이딩 윈도우로 전처리 O / LLM O / Stable Audio 기본, LoRA X
    - Model4: 슬라이딩 윈도우로 전처리 O / LLM X / Stable Audio 기본, LoRA O (이때 LoRA는 Model3에서 생성된 프롬프트를 보고 인간이 rating한 것을 참고, 즉 LLM은 사용하긴 하지만 모델이 쓰는 건 프롬프트가 아니라 인간이 rating한 것.)
5. **출력**
    - Ebook 슬라이딩 윈도우마다 대응되는 짧은 BGM 오디오





전처리 + LoRA 파인튜닝한 Stable Audio 를 만들 건데,

우선 1. model4_lora_prompts_short(이미 Model3대로 전처리, LLM프롬프트로 나온 것)를 Model3_generate2.py(기본 LoRA 없는 Baseline Stable Audio)에 넣어서 만들고 model4_lora_audio 폴더에 저장
2. 그 오디오를 model4_lora_prompts.csv 파일에 있는 텍스트만 골라서
3. 왜 적절한지/부적절한지 이유와 5점 척도를 rating해서 (1명만)
4. 그걸 lora 모델로, Model4 이름으로 빌드하고
5. Model4에서 prompts.json에 있는 3번째 프롬프트로 만들고 lora_outputs 폴더에 저장
6. 그걸 Model3에서 만든 3번째 오디오랑 비교하기(LoRA 하기 전/후를 비교해서 LoRA가 의미있었는지 확인하기 위함)
7. 추가로 그 부분에 해당하는 baseline도 Model5로 만들어서 3,4,5 비교하기
