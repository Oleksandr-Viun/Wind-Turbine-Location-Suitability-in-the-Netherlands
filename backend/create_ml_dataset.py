import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
from scipy.spatial import cKDTree
import numpy as np
from pathlib import Path

def build_machine_learning_dataset():
    # Настраиваем относительные пути
    base_dir = Path("data/processed")
    raw_dir = Path("data/raw")
    
    print("1. Загружаю базовую сетку ветра (wind_grid_final.csv)...")
    df_grid = pd.read_csv(base_dir / "wind_grid_final.csv")
    grid_geom = [Point(xy) for xy in zip(df_grid['cell_lon'], df_grid['cell_lat'])]
    # Наша сетка сейчас в градусах (EPSG:4326)
    gdf_grid = gpd.GeoDataFrame(df_grid, geometry=grid_geom, crs="EPSG:4326")
    
    # Для точных метрических вычислений переводим копию сетки в голландские метры (RD New)
    gdf_grid_rd = gdf_grid.to_crs(epsg=28992)

    # =========================================================
    # ФИЧА 1: ОГРАНИЧЕНИЯ NATURA 2000
    # =========================================================
    print("2. Обрабатываю зоны Natura 2000...")
    df_natura = pd.read_csv(raw_dir / "natura2000_pdok.csv")
    df_natura['geometry'] = df_natura['geometry_wkt'].apply(wkt.loads)
    # Natura 2000 у нас в градусах (EPSG:4326)
    gdf_natura = gpd.GeoDataFrame(df_natura, geometry='geometry', crs="EPSG:4326")
    
    # Проверка "Точка в полигоне"
    joined_natura = gpd.sjoin(gdf_grid, gdf_natura, how="left", predicate="within")
    
    # Если точка попала в парк, будет 1, иначе 0
    df_grid['is_natura2000'] = (~joined_natura['index_right'].isna()).astype(int)
    print(f"   -> Найдено {df_grid['is_natura2000'].sum()} точек сетки внутри заповедников.")

    # =========================================================
    # ФИЧА 2: СУЩЕСТВУЮЩИЕ ВЕТРЯКИ (Расстояние в метрах)
    # =========================================================
    print("3. Вычисляю расстояние до ближайших существующих ветряков...")
    df_turbines = pd.read_csv(raw_dir / "windturbines_rivm_ashoogte.csv")
    df_turbines['geometry'] = df_turbines['geometry_wkt'].apply(wkt.loads)
    
    # ВАЖНО: Ветряки изначально записаны в голландских метрах (EPSG:28992)
    gdf_turbines = gpd.GeoDataFrame(df_turbines, geometry='geometry', crs="EPSG:28992")
    
    # Извлекаем X, Y (в метрах) для обеих карт
    grid_coords_m = np.array(list(zip(gdf_grid_rd.geometry.x, gdf_grid_rd.geometry.y)))
    turbine_coords_m = np.array(list(zip(gdf_turbines.geometry.x, gdf_turbines.geometry.y)))
    
    # Используем KD-Tree для мгновенного поиска ближайшего соседа
    tree = cKDTree(turbine_coords_m)
    distances, _ = tree.query(grid_coords_m, k=1)
    
    # Записываем расстояние в метрах, округленное до целых чисел
    df_grid['dist_to_nearest_turbine_m'] = np.round(distances).astype(int)

    # =========================================================
    # ФИЧА 3: ПЛОТНОСТЬ НАСЕЛЕНИЯ (Муниципалитеты)
    # =========================================================
    print("4. Пришиваю плотность населения из нового файла .gpkg...")
    # Загружаем склеенный файл, который ты сделал на предыдущем шаге
    gdf_mun = gpd.read_file(base_dir / "municipality_full.gpkg")
    
    # Убеждаемся, что система координат совпадает с сеткой (EPSG:4326)
    gdf_mun = gdf_mun.to_crs(epsg=4326)
    
    # Проверка "Точка в полигоне"
    joined_mun = gpd.sjoin(gdf_grid, gdf_mun, how="left", predicate="within")
    
    # Забираем плотность. Если точка в море (NaN), ставим 0
    df_grid['population_density'] = joined_mun['population_density'].fillna(0).astype(int)

    # =========================================================
    # ФИНАЛИЗАЦИЯ И СОХРАНЕНИЕ
    # =========================================================
    print("5. Очищаю таблицу и сохраняю ML-датасет...")
    # Оставляем только нужные колонки, убираем технический мусор
    final_columns = [
        'cell_lon', 
        'cell_lat', 
        'wind_speed', 
        'is_natura2000', 
        'dist_to_nearest_turbine_m', 
        'population_density'
    ]
    df_ml = df_grid[final_columns]
    
    output_file = base_dir / "ml_dataset_final.csv"
    df_ml.to_csv(output_file, index=False)
    print(f"🎉 УСПЕХ! Полный датасет для обучения модели сохранен: {output_file}")
    
    # Показываем превью первых трех строчек
    print("\nПревью ваших данных:")
    print(df_ml.head(3).to_string())

if __name__ == "__main__":
    build_machine_learning_dataset()