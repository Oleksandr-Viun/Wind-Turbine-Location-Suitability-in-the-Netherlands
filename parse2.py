import requests
import pandas as pd
from pathlib import Path

def download_and_read_knmi_data():
    # 1. Создаем правильную структуру папок для проекта (как принято в Data Science)
    # Создаст папку data, а внутри нее raw, если их еще нет
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = output_dir / "knmi_daily_wind_2015_2024.csv"

    # 2. Настраиваем запрос к API
    url = "https://www.daggegevens.knmi.nl/klimatologie/daggegevens"
    
    # Payload - это параметры нашего запроса
    payload = {
        "start": "20150101",  # Начало: 1 января 2015
        "end": "20241231",    # Конец: 31 декабря 2024 (ровно 10 лет данных)
        "vars": "DDVEC:FHVEC:FG:FHX:FXX", # Запрашиваем только параметры ветра
        "stns": "ALL",        # Со всех станций Нидерландов
        "fmt": "csv",         # Формат вывода: CSV (идеально для pandas)
    }

    print("Отправляю запрос на сервер KNMI... (подожди 10-30 секунд)")
    
    # 3. Делаем POST-запрос
    try:
        response = requests.post(url, data=payload, timeout=120)
        # Если сервер вернул ошибку (например, 404 или 500), скрипт остановится и покажет её
        response.raise_for_status() 
        
        # 4. Сохраняем полученный текст в CSV-файл
        file_path.write_text(response.text, encoding="utf-8")
        print(f"✅ Успех! Датасет сохранен сюда: {file_path}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка при скачивании: {e}")
        return

    # ==============================================================
    # ШАГ 2: СРАЗУ ОТКРЫВАЕМ ФАЙЛ В PANDAS (Data Exploration)
    # ==============================================================
    print("\n--- Чтение данных в Pandas ---")
    
    # ВАЖНО: У KNMI в начале файла идут десятки строк с описанием (метаданные), 
    # они начинаются со знака '#'. Параметр comment='#' заставляет pandas их игнорировать 
    # и читать только саму таблицу с цифрами.
    # skipinitialspace=True убирает лишние пробелы после запятых.
    df = pd.read_csv(file_path, comment='#', skipinitialspace=True)
    
    print(f"Всего загружено строк: {len(df)}")
    print("\nПервые 5 строк твоего готового датасета:")
    print(df.head())
    
    print("\nСписок колонок:")
    print(df.columns.tolist())

# Запускаем функцию
if __name__ == "__main__":
    download_and_read_knmi_data()