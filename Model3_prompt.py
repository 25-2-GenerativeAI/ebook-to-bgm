import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# -------------------------------
# 환경 설정
# -------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("🚨 OPENAI_API_KEY not found in .env file")

client = OpenAI()

# -------------------------------
# 프롬프트 생성 함수
# -------------------------------
def generate_prompt(emotion: str, text: str, senses: list[str] = None) -> str:
    senses = senses or []
    sense_str = ", ".join(senses) if senses else "none"

    system_msg = """You are a soundtrack designer for an audio diffusion model.
Analyze the text excerpt, focusing on:
1. Overall emotional tone (joy, sadness, fear, tension, wonder, etc.)
2. Narrative atmosphere (mystical, romantic, tragic, suspenseful, epic, etc.)
3. Suitable instrumentation and sound textures (piano, strings, ambient drones, percussion, electronic pulses, natural sounds, etc.)

Output: A single concise prompt (one sentence) in English,
describing the soundtrack that best matches the scene,
optimized for audio generation.
"""

    user_msg = f"""
Emotion: {emotion}
Senses: {sense_str}
Text excerpt: {text[:300]}...
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
        return response.choices[0].message.content.strip()
    return "(no response)"

# -------------------------------
# 실행부
# -------------------------------
def main(input_json="scene_analysis.json", output_json="prompts.json"):
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"{input_json} not found.")

    with open(input_json, "r", encoding="utf-8") as f:
        scenes = json.load(f)

    prompts = []
    for scene in scenes:
        emotion = scene["dominant_emotion"]
        text_excerpt = scene["text"]

        prompt = generate_prompt(emotion, text_excerpt)
        print(f"\n🎬 Prompt generated:\n{prompt}\n")

        prompts.append(prompt)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)

    print(f"✅ Prompts saved to {output_json}")

if __name__ == "__main__":
    main()
