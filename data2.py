import requests
import pandas as pd
from io import StringIO
from pathlib import Path

# 1. Создаем папку data/raw
output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)
file_path = output_dir / "knmi_daily_wind_2015_2024.csv"

# 2. Твой API и параметры (я убрал "fmt": "csv", чтобы сервер не путался)
url = "https://www.daggegevens.knmi.nl/klimatologie/daggegevens"
payload = {
    "start": "20350101",
    "end": "20241231",
    "vars": "DDVEC:FHVEC:FG:FHX:FXX",
    "stns": "ALL"
}

# КРИТИЧЕСКИ ВАЖНО: Притворяемся браузером, чтобы KNMI не отдал нам HTML-заглушку
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

print("Скачиваю данные с daggegevens.knmi.nl... (ожидание ~5-15 сек)")
response = requests.post(url, data=payload, headers=headers, timeout=120)
response.raise_for_status()

text_data = response.text

# 3. Проверяем, не HTML ли это
if "<html" in text_data.lower() or "<!doctype" in text_data.lower():
    print("❌ ОШИБКА: Сервер снова вернул HTML. Сохраняю как .html для проверки.")
    file_path.with_suffix('.html').write_text(text_data, encoding="utf-8")
else:
    # 4. Сохраняем чистые данные
    file_path.write_text(text_data, encoding="utf-8")
    print(f"✅ Данные успешно скачаны: {file_path}")
    
    # 5. Обработка для Pandas (Data Preparation)
    # Так как оригинальные данные KNMI разделены запятыми, но имеют кучу '#',
    # мы жестко задаем колонки, чтобы избежать ошибки "ParserError"
    col_names = ["STN", "YYYYMMDD", "DDVEC", "FHVEC", "FG", "FHX", "FXX"]
    
    df = pd.read_csv(
        StringIO(text_data), 
        comment='#',               # Игнорируем весь текстовый мусор от KNMI
        names=col_names,           # Принудительно ставим названия колонок
        skipinitialspace=True      # Убираем возможные пробелы после запятых
    )
    
    # Переводим ветер из 0.1 м/с в нормальные метры в секунду
    for col in ["FHVEC", "FG", "FHX", "FXX"]:
        if col in df.columns:
            df[col] = df[col] / 10.0
            
    print("\n=========================================================")
    print(" ГОТОВЫЙ РЕЗУЛЬТАТ: ВЕТЕР В НИДЕРЛАНДАХ (2015-2024) м/с")
    print("=========================================================")
    print(df.head(15).to_string(index=False))
    print(f"\nВсего загружено строк со всех метеостанций: {len(df)}")