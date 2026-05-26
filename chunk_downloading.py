import requests
import pandas as pd
from io import StringIO
from pathlib import Path
import time

def download_historical_wind_data():
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    final_file_path = output_dir / "knmi_wind_1995_2024.csv"

    url = "https://www.daggegevens.knmi.nl/klimatologie/daggegevens"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    col_names = ["STN", "YYYYMMDD", "DDVEC", "FHVEC", "FG", "FHX", "FXX"]

    start_year = 1995
    end_year = 2024
    chunk_size = 5  # Качаем по 5 лет за раз

    all_dataframes = []

    print(f"Начинаю скачивание данных с {start_year} по {end_year} год чанками...\n")

    # Идем циклом с шагом в 5 лет
    for y in range(start_year, end_year + 1, chunk_size):
        # Формируем даты начала и конца чанка
        current_end_year = min(y + chunk_size - 1, end_year)
        date_start = f"{y}0101"
        date_end = f"{current_end_year}1231"
        
        print(f"📥 Запрашиваю период: {y} - {current_end_year}...")
        
        payload = {
            "start": date_start,
            "end": date_end,
            "vars": "DDVEC:FHVEC:FG:FHX:FXX",
            "stns": "ALL"
        }

        try:
            response = requests.post(url, data=payload, headers=headers, timeout=120)
            response.raise_for_status()
            text_data = response.text

            # Защита от HTML-заглушек
            if "<html" in text_data.lower():
                print(f"❌ ОШИБКА: Сервер вернул HTML для периода {y}-{current_end_year}. Пропускаю.")
                continue

            # Читаем чанк в Pandas
            chunk_df = pd.read_csv(
                StringIO(text_data), 
                comment='#', 
                names=col_names, 
                skipinitialspace=True
            )
            
            all_dataframes.append(chunk_df)
            print(f"✅ Успешно получено строк: {len(chunk_df)}")

            # Спим 2 секунды, чтобы быть "вежливыми" к серверу KNMI
            time.sleep(2)

        except Exception as e:
            print(f"❌ Ошибка соединения на периоде {y}-{current_end_year}: {e}")

    # ==========================================
    # ОБЪЕДИНЕНИЕ И ОЧИСТКА ДАННЫХ
    # ==========================================
    print("\nСклеиваю все чанки воедино...")
    if not all_dataframes:
        print("Не удалось скачать ни одного чанка :(")
        return

    # Склеиваем все куски в одну большую таблицу
    final_df = pd.concat(all_dataframes, ignore_index=True)

    # Переводим ветер из 0.1 м/с в нормальные метры в секунду
    for col in ["FHVEC", "FG", "FHX", "FXX"]:
        if col in final_df.columns:
            final_df[col] = final_df[col] / 10.0

    # Сохраняем финальный результат
    final_df.to_csv(final_file_path, index=False)
    
    print("\n=========================================================")
    print(f"🎉 ГОТОВО! Сохранен файл: {final_file_path}")
    print(f"📊 Всего собрано записей: {len(final_df)}")
    print(f"📅 Период: с {final_df['YYYYMMDD'].min()} по {final_df['YYYYMMDD'].max()}")
    print("=========================================================")
    
    # Покажем кусочек данных
    print(final_df.head())

if __name__ == "__main__":
    download_historical_wind_data()