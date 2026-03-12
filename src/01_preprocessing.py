import pandas as pd
from pathlib import Path

INPUT_PATH = Path('data/data_clean.csv')
OUTPUT_PATH = Path('data/processed_base.csv')

SCORE_MAPPING = {
    'До 70': 1,
    '70-80': 2,
    '80-90': 3,
    '90+': 4,
}

TIME_MAPPING = {
    'Первая половина 10го класса': 0,
    'Вторая половина 10го класса': 1,
    'Первая половина 11го класса': 2,
    'Вторая половина 11го класса': 3,
    'Интенсив': 4,
    'Я считаю что сдам своими силами': 6,
}

def normalize_school(school_name: str) -> int:
    if pd.isna(school_name):
        return 0
    return 1 if 'Лицей БГУ' in str(school_name) else 0

def count_tutor_subjects(subjects_str: str) -> int:
    if pd.isna(subjects_str) or str(subjects_str).strip() == '':
        return 0
    return len([s.strip() for s in str(subjects_str).split(',') if s.strip()])

def parse_mapping(val: str, mapping: dict, default_val: int) -> int:
    if pd.isna(val):
        return default_val
    return mapping.get(str(val).strip(), default_val)

def calculate_time_lag(thought: int, enrolled: int) -> int:
    return max(0, enrolled - thought)

def pipeline():
    df = pd.read_csv(INPUT_PATH)
    
    df['School_Type'] = df['Где вы учитесь?'].apply(normalize_school)
    
    df['Score_Level'] = df['На какой балл по экзу ты реально рассчитываешь?'].apply(
        lambda x: parse_mapping(x, SCORE_MAPPING, 2)
    ) 
    
    df['Enrolled_Time'] = df['Когда ты по итогу записался на курсы / взял репета?'].apply(
        lambda x: parse_mapping(x, TIME_MAPPING, 2)
    )
    df['Thought_Time'] = df['Когда ты впервые задумался о том чтобы пойти на курсы/взять репета?'].apply(
        lambda x: parse_mapping(x, TIME_MAPPING, 2)
    )
    
    df['Time_Lag'] = df.apply(lambda row: calculate_time_lag(row['Thought_Time'], row['Enrolled_Time']), axis=1)
    df['Num_Tutor_Subjects'] = df['По каким предметам ты взял репета?'].apply(count_tutor_subjects)
    
    final_columns = ['School_Type', 'Score_Level', 'Num_Tutor_Subjects', 'Time_Lag', 'Enrolled_Time']
    df_final = df[final_columns].copy()
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"Файл сохранен в {OUTPUT_PATH}")

if __name__ == "__main__":
    pipeline()