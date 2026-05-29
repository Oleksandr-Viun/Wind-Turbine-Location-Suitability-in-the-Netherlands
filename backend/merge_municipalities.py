import pandas as pd
import geopandas as gpd

def merge_municipality_data():
    # ИСПРАВЛЕННЫЕ ПУТИ:
    csv_file = "data/raw/clean_cbs_municipalities.csv"
    geojson_file = "data/raw/gemeente_location.geojson"
    output_file = "data/processed/municipality_full.gpkg"

    print("1. Загружаю CSV с данными о населении...")
    df_stats = pd.read_csv(csv_file)
    
    print("2. Загружаю GeoJSON с полигонами границ (это может занять пару секунд)...")
    gdf_borders = gpd.read_file(geojson_file)
    
    print("3. Склеиваю геометрию и статистику по ID муниципалитета...")
    # Делаем Inner Join: оставляем только те муниципалитеты, которые есть в обоих файлах
    gdf_merged = gdf_borders.merge(
        df_stats, 
        left_on='statcode', 
        right_on='municipality_code', 
        how='inner'
    )
    
    print(f"   ✅ Успешно склеено {len(gdf_merged)} муниципалитетов.")
    
    # 4. Очистка. Оставляем только самое важное, чтобы не раздувать файл
    cols_to_keep = [
        'municipality_code', 
        'municipality_name', 
        'population_total', 
        'population_density', 
        'urbanity_category', 
        'geometry' 
    ]
    gdf_final = gdf_merged[cols_to_keep]
    
    print("5. Сохраняю результат в сверхбыстрый формат GeoPackage (.gpkg)...")
    # Переводим в систему координат GPS
    gdf_final = gdf_final.to_crs(epsg=4326) 
    gdf_final.to_file(output_file, driver="GPKG")
    
    print(f"🎉 ГОТОВО! Ваш новый идеальный файл: {output_file}")

if __name__ == "__main__":
    merge_municipality_data()