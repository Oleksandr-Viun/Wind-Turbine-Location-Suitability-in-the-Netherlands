import pandas as pd
from pymongo import MongoClient
from pathlib import Path

def upload_to_mongodb():
    # 1. Подключение к MongoDB (замени URL, если используешь Atlas)
    client = MongoClient("mongodb+srv://admin_user:jIdclpFqQhOVvhHb@cluster0.346pghw.mongodb.net/?appName=Cluster0")
    db = client["wind_project"]  # Создаем/выбираем базу данных
    collection = db["grid"]      # Создаем/выбираем коллекцию (таблицу)

    # Очищаем старые данные, если скрипт запускается не первый раз
    collection.drop()

    # 2. Читаем наш финальный CSV
    csv_path = Path("data/processed/ml_dataset_final.csv")
    print(f"Читаю файл {csv_path}...")
    df = pd.read_csv(csv_path)

    # 3. Преобразуем данные в формат документов MongoDB + GeoJSON
    records = []
    for _, row in df.iterrows():
        doc = {
            "wind_speed": float(row["wind_speed"]),
            "is_natura2000": int(row["is_natura2000"]),
            "dist_to_nearest_turbine_m": int(row["dist_to_nearest_turbine_m"]),
            "population_density": int(row["population_density"]),
            # СПЕЦИАЛЬНЫЙ ФОРМАТ GEOJSON ДЛЯ БЫСТРОГО ПОИСКА
            "location": {
                "type": "Point",
                "coordinates": [float(row["cell_lon"]), float(row["cell_lat"])] # ВАЖНО: [Долгота, Широта]
            }
        }
        records.append(doc)

    # 4. Массовая загрузка (Bulk Insert)
    print("Загружаю данные в MongoDB...")
    collection.insert_many(records)

    # 5. МАГИЯ: Создаем пространственный индекс (R-Tree под капотом MongoDB)
    collection.create_index([("location", "2dsphere")])

    print(f"✅ Успешно загружено {len(records)} точек в MongoDB!")
    print("✅ Создан гео-индекс 2dsphere.")

if __name__ == "__main__":
    upload_to_mongodb()
