import pandas as pd
import requests
from pathlib import Path

def merge_coordinates_to_dataset():
    # Пути к файлам
    raw_file = Path("data/raw/knmi_wind_1995_2024.csv")
    
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    output_full = processed_dir / "knmi_wind_daily_with_coords.csv"
    output_summary = processed_dir / "knmi_stations_summary.csv"

    if not raw_file.exists():
        print("❌ Файл с 30-летними данными не найден!")
        return

    print("1. Загружаю твой исторический датасет (это может занять пару секунд)...")
    df_raw = pd.read_csv(raw_file)
    existing_stns = df_raw['STN'].unique().tolist()

    print("2. Запрашиваю координаты с сервера KNMI (хитрость с 1 днем)...")
    url = "https://www.daggegevens.knmi.nl/klimatologie/daggegevens"
    payload = {"start": "20240101", "end": "20240101", "vars": "FG", "stns": "ALL"}
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.post(url, data=payload, headers=headers)
    text_data = response.text

    print("3. Извлекаю широту и долготу...")
    stations_meta = []
    
    for line in text_data.split('\n'):
        if line.startswith('#'):
            parts = line.replace('#', '').strip().split()
            if len(parts) >= 5 and parts[0].replace(':', '').isdigit():
                try:
                    stn_id = int(parts[0].replace(':', ''))
                    lon = float(parts[1])
                    lat = float(parts[2])
                    name = " ".join(parts[4:]) 
                    
                    if stn_id in existing_stns:
                        stations_meta.append({
                            "STN": stn_id,
                            "station_name": name,
                            "lat": lat,
                            "lon": lon
                        })
                except ValueError:
                    continue

    df_geo = pd.DataFrame(stations_meta)
    
    if df_geo.empty:
        print("❌ Ошибка парсинга координат. Проверь ответ сервера.")
        return

    print("\n4. Выполняю слияние (MERGE) таблиц...")
    
    # ---------------------------------------------------------
    # РЕЗУЛЬТАТ 1: Полный ежедневный датасет с координатами
    # ---------------------------------------------------------
    # Приклеиваем lat, lon и name ко всем 1.3 миллионам строк
    df_full_merged = pd.merge(df_raw, df_geo, on="STN", how="inner")
    
    # Переставляем колонки для красоты (чтобы гео-данные были в начале)
    cols = ['STN', 'station_name', 'lat', 'lon', 'YYYYMMDD', 'DDVEC', 'FHVEC', 'FG', 'FHX', 'FXX']
    df_full_merged = df_full_merged[cols]
    
    df_full_merged.to_csv(output_full, index=False)
    print(f"✅ Полный датасет сохранен: {output_full}")

    # ---------------------------------------------------------
    # РЕЗУЛЬТАТ 2: Компактный Summary для карты 200x200
    # ---------------------------------------------------------
    print("5. Рассчитываю средний многолетний ветер для карты...")
    # Считаем среднее для каждой станции
    df_wind_avg = df_raw.groupby("STN")[["FG", "FHVEC", "FHX", "FXX"]].mean().reset_index()
    
    # Объединяем средние показатели с координатами
    df_summary = pd.merge(df_geo, df_wind_avg, on="STN", how="inner")
    
    # Переименовываем колонки в понятный формат (м/с)
    df_summary = df_summary.rename(columns={
        "FG": "avg_wind_speed",
        "FHVEC": "vector_wind_speed",
        "FHX": "max_hourly_wind",
        "FXX": "max_gust_speed"
    })
    
    df_summary.to_csv(output_summary, index=False)
    print(f"✅ Датасет для интерполяции сохранен: {output_summary}")

    print("\n=========================================================")
    print(" ГОТОВО! ЭТАП СБОРА И ОЧИСТКИ ДАННЫХ (DATA PREP) ЗАВЕРШЕН")
    print("=========================================================")
    print("Пример того, что получилось (Summary):")
    print(df_summary[['STN', 'station_name', 'lat', 'lon', 'avg_wind_speed']].head())

if __name__ == "__main__":
    merge_coordinates_to_dataset()