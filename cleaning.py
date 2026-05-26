import pandas as pd
from pathlib import Path

def clean_wind_data():
    input_file = Path("data/processed/knmi_wind_daily_with_coords.csv")
    output_file = Path("data/processed/knmi_wind_clean.csv")
    
    if not input_file.exists():
        print("❌ Файл не найден. Сначала выполни предыдущий скрипт.")
        return

    print("Загружаю сырой датасет...")
    df = pd.read_csv(input_file)
    initial_rows = len(df)
    
    print(f"Изначальное количество строк: {initial_rows}")
    print("\nНачинаю очистку данных...")

    # ПРАВИЛО 1: Форматирование дат
    # Превращаем число 20240101 в настоящую дату 2024-01-01
    df['date'] = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d')
    df = df.drop(columns=['YYYYMMDD']) # Удаляем старую колонку

    # ПРАВИЛО 2: Удаление пропусков (NaN)
    # Если нет данных о среднем ветре или порывах - эта строка нам не нужна
    df_clean = df.dropna(subset=['FG', 'FHX', 'FXX', 'DDVEC']).copy()
    nan_dropped = initial_rows - len(df_clean)

    # ПРАВИЛО 3: Физические ограничения (Sanity Checks)
    # 3.1 Ветер не может быть отрицательным
    df_clean = df_clean[df_clean['FG'] >= 0]
    
    # 3.2 Направление ветра должно быть от 0 до 360 градусов
    df_clean = df_clean[(df_clean['DDVEC'] >= 0) & (df_clean['DDVEC'] <= 360)]
    
    # 3.3 Порыв ветра всегда больше или равен средней скорости
    df_clean = df_clean[df_clean['FXX'] >= df_clean['FG']]
    
    # 3.4 Исключаем нереалистичные аномалии (например, скорость > 60 м/с)
    # Исторический максимум в Нидерландах был около 40-45 м/с.
    df_clean = df_clean[df_clean['FXX'] < 60]

    final_rows = len(df_clean)
    physics_dropped = (initial_rows - nan_dropped) - final_rows

    # ПЕРЕСТАНОВКА КОЛОНОК (для удобства)
    cols = ['STN', 'station_name', 'lat', 'lon', 'date', 'DDVEC', 'FHVEC', 'FG', 'FHX', 'FXX']
    df_clean = df_clean[cols]

    # Сохраняем чистый датасет
    df_clean.to_csv(output_file, index=False)

    # ---------------------------------------------------------
    # ОТЧЕТ ОБ ОЧИСТКЕ (DATA QUALITY REPORT)
    # ---------------------------------------------------------
    print("\n==================================================")
    print(" 🧹 ОТЧЕТ ОБ ОЧИСТКЕ ДАННЫХ (DATA CLEANING REPORT)")
    print("==================================================")
    print(f"Всего строк до очистки:        {initial_rows}")
    print(f"Удалено из-за пустых значений: {nan_dropped} строк")
    print(f"Удалено из-за ошибок физики:   {physics_dropped} строк")
    print("-" * 50)
    print(f"Итоговых чистых строк:         {final_rows}")
    print(f"Процент сохраненных данных:    {round((final_rows/initial_rows)*100, 2)}%")
    print("==================================================")
    print(f"✅ Чистый файл сохранен: {output_file}")

if __name__ == "__main__":
    clean_wind_data()