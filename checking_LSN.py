import pandas as pd

def load_sensorimotor_db(csv_path: str) -> dict:
    """
    Lancaster Sensorimotor Norms CSV 불러오기
    반환: {단어: {감각: 점수}}
    """
    df = pd.read_csv(csv_path)

    # 컬럼 이름 정리
    sense_map = {
        "Auditory.mean": "hearing",
        "Gustatory.mean": "taste",
        "Haptic.mean": "touch",
        "Interoceptive.mean": "interoception",
        "Olfactory.mean": "smell",
        "Visual.mean": "vision"
    }

    db = {}
    for _, row in df.iterrows():
        word = str(row["Word"]).lower()
        db[word] = {
            sense_map[col]: float(row[col]) 
            for col in sense_map if pd.notna(row[col])
        }
    return db

# ---------------- 실행 예시 ----------------
if __name__ == "__main__":
    SENSORIMOTOR_DB = load_sensorimotor_db("LSN.csv")
    print("총 단어 수:", len(SENSORIMOTOR_DB))
    print("apple →", SENSORIMOTOR_DB.get("apple"))
    print("music →", SENSORIMOTOR_DB.get("music"))
