import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from pathlib import Path

def generate_perfect_wind_map():
    input_file = Path("data/processed/knmi_stations_summary.csv")
    output_file = Path("data/processed/wind_grid_final.csv")
    
    # 1. ЗАГРУЗКА ДАННЫХ И ИНТЕРПОЛЯЦИЯ (Снова делаем плавно)
    print("1. Рассчитываю плавную карту ветра (Rbf)...")
    df_stations = pd.read_csv(input_file)
    x_st = df_stations['lon'].values
    y_st = df_stations['lat'].values
    z_wind = df_stations['avg_wind_speed'].values

    lon_min, lon_max = x_st.min() - 0.3, x_st.max() + 0.3
    lat_min, lat_max = y_st.min() - 0.3, y_st.max() + 0.3

    grid_lon = np.linspace(lon_min, lon_max, 200)
    grid_lat = np.linspace(lat_min, lat_max, 200)
    X_grid, Y_grid = np.meshgrid(grid_lon, grid_lat)

    rbf = Rbf(x_st, y_st, z_wind, function='inverse', power=2)
    Z_grid = rbf(X_grid, Y_grid)

    # Упаковываем сетку 200x200 в GeoDataFrame
    df_grid = pd.DataFrame({
        "cell_lon": X_grid.flatten(),
        "cell_lat": Y_grid.flatten(),
        "wind_speed": Z_grid.flatten()
    })
    grid_geom = [Point(xy) for xy in zip(df_grid['cell_lon'], df_grid['cell_lat'])]
    gdf_grid = gpd.GeoDataFrame(df_grid, geometry=grid_geom, crs="EPSG:4326")

    # =========================================================
    # 2. УМНОЕ МАСКИРОВАНИЕ (ПО ТВОЕЙ ИДЕЕ)
    # =========================================================
    print("2. Загружаю официальные границы суши (JSON)...")
    url_geojson = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries/NLD.geo.json"
    nl_land = gpd.read_file(url_geojson)

    print("3. Ищу морские станции и создаю для них буферные зоны...")
    # Превращаем станции в GeoDataFrame
    st_geom = [Point(xy) for xy in zip(x_st, y_st)]
    gdf_stations = gpd.GeoDataFrame(df_stations, geometry=st_geom, crs="EPSG:4326")

    # Ищем, какие станции лежат НА СУШЕ (внутри nl_land)
    stations_on_land = gpd.sjoin(gdf_stations, nl_land, how="inner", predicate="within")
    
    # Морские станции - это те, которых нет в списке сухопутных
    sea_stations = gdf_stations[~gdf_stations['STN'].isin(stations_on_land['STN'])].copy()
    
    print(f"   Найдено морских станций: {len(sea_stations)}. Растягиваю море вокруг них...")
    # Делаем буфер ~25 км (0.25 градуса) ТОЛЬКО для морских станций
    sea_buffers = sea_stations.copy()
    sea_buffers['geometry'] = sea_buffers.geometry.buffer(0.25)

    print("4. Склеиваю сушу и морские зоны в единую маску...")
    # Объединяем геометрию суши и морских кругов
    combined_mask = pd.concat([nl_land[['geometry']], sea_buffers[['geometry']]])
    # dissolve() сливает все пересекающиеся фигуры в один сплошной контур
    combined_mask = combined_mask.dissolve()

    print("5. Вырезаю финальную карту...")
    gdf_final = gpd.sjoin(gdf_grid, combined_mask, how="inner", predicate="within")

    # Сохраняем результат
    gdf_final[['cell_lon', 'cell_lat', 'wind_speed']].to_csv(output_file, index=False)
    print(f"✅ УСПЕХ! Идеальная сетка сохранена: {output_file}")

    # =========================================================
    # 3. ВИЗУАЛИЗАЦИЯ
    # =========================================================
    print("Рисую результат...")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Рисуем нашу склеенную маску-границу (Красная линия)
    combined_mask.boundary.plot(ax=ax, color='red', linewidth=1.5, label="Финальный контур")
    
    # Рисуем интерполированный ветер
    scatter = ax.scatter(
        gdf_final['cell_lon'], gdf_final['cell_lat'], 
        c=gdf_final['wind_speed'], cmap='YlGnBu', s=15, alpha=0.9
    )
    
    # Точки станций
    ax.scatter(x_st, y_st, color='black', s=10, label="Метеостанции")
    
    plt.colorbar(scatter, label='Средняя скорость ветра (м/с)')
    plt.title("Идеальная карта ветра (Суша + Морские зоны)")
    plt.xlabel("Долгота")
    plt.ylabel("Широта")
    plt.legend()
    
    plt.savefig("data/processed/wind_map_perfect.png")
    plt.show()

if __name__ == "__main__":
    generate_perfect_wind_map()