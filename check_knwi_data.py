import pandas as pd
import requests
from pathlib import Path

def fetch_and_merge_exact_coordinates():
    raw_file = Path("data/raw/knmi_wind_1995_2024.csv")
    output_file = Path("data/processed/exact_stations_summary.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not raw_file.exists():
        print("❌ Файл с 30-летними данными не найден!")
        return

    print("1. Анализируем твой датасет...")
    df_raw = pd.read_csv(raw_file)
    existing_stns = df_raw['STN'].unique().tolist()
    print(f"В твоем датасете найдено {len(existing_stns)} уникальных станций.")

    print("\n2. Скачиваю официальные метаданные напрямую с KNMI...")
    url = "https://www.daggegevens.knmi.nl/klimatologie/daggegevens"
    payload = {
        "start": "20240101", "end": "20240101", 
        "vars": "FG", "stns": "ALL"
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.post(url, data=payload, headers=headers)
    text_data = response.text

    print("3. Извлекаю координаты...\n")
    stations_meta = []
    
    # Новый, супер-надежный парсер
    for line in text_data.split('\n'):
        if line.startswith('#'):
            # Убираем решетку и разбиваем строку на куски по пробелам
            parts = line.replace('#', '').strip().split()
            
            # Проверяем, что в строке достаточно данных (минимум 5 колонок) 
            # и что первое слово - это цифры (номер станции)
            if len(parts) >= 5 and parts[0].replace(':', '').isdigit():
                try:
                    # Извлекаем данные
                    stn_id = int(parts[0].replace(':', ''))
                    lon = float(parts[1])
                    lat = float(parts[2])
                    # Все остальные слова склеиваем в название станции
                    name = " ".join(parts[4:]) 
                    
                    if stn_id in existing_stns:
                        stations_meta.append({
                            "STN": stn_id,
                            "station_name": name,
                            "lat": lat,
                            "lon": lon
                        })
                except ValueError:
                    continue # Если не удалось превратить в цифру, пропускаем строку

    df_geo = pd.DataFrame(stations_meta)
    
    # ЗАЩИТА ОТ ОШИБКИ: Если таблица всё еще пустая, печатаем сырой текст
    if df_geo.empty:
        print("❌ ОШИБКА: Не удалось найти координаты! Вот что прислал сервер KNMI:")
        print(text_data[:1500]) # Печатаем первые 1500 символов
        return

    print("-" * 60)
    print(" НАЙДЕННЫЕ КООРДИНАТЫ ДЛЯ ТВОИХ СТАНЦИЙ")
    print("-" * 60)
    print(df_geo.head(10).to_string(index=False)) # Выводим первые 10 для проверки
    print(f"...и еще {len(df_geo) - 10} станций.")
    print("-" * 60)

    print("\n4. Усредняю ветер за 30 лет и объединяю с координатами...")
    df_wind_avg = df_raw.groupby("STN")[["FG", "FHVEC", "FHX", "FXX"]].mean().reset_index()
    
    df_final = pd.merge(df_geo, df_wind_avg, on="STN", how="inner")
    
    df_final = df_final.rename(columns={
        "FG": "avg_wind_speed",
        "FHVEC": "vector_wind_speed",
        "FHX": "max_hourly_wind",
        "FXX": "max_gust_speed"
    })

    df_final.to_csv(output_file, index=False)
    print(f"\n✅ ГОТОВО! Идеальный датасет сохранен в: {output_file}")
    print(f"Всего станций сохранено: {len(df_final)}")

if __name__ == "__main__":
    fetch_and_merge_exact_coordinates()