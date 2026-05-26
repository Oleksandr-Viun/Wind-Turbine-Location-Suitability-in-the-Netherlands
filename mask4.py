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
    
    # ---------------------------------------------------------
    # ШАГ 1: ИНТЕРПОЛЯЦИЯ ВЕТРА (Математика Rbf)
    # ---------------------------------------------------------
    print("1. Рассчитываю математическую модель ветра...")
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

    df_grid = pd.DataFrame({
        "cell_lon": X_grid.flatten(),
        "cell_lat": Y_grid.flatten(),
        "wind_speed": Z_grid.flatten()
    })
    grid_geom = [Point(xy) for xy in zip(df_grid['cell_lon'], df_grid['cell_lat'])]
    gdf_grid = gpd.GeoDataFrame(df_grid, geometry=grid_geom, crs="EPSG:4326")

    # ---------------------------------------------------------
    # ШАГ 2: ТОЧЕЧНАЯ МАСКА (Суша + Вода у Гааги)
    # ---------------------------------------------------------
    print("2. Загружаю идеальную карту суши...")
    url_geojson = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
    world = gpd.read_file(url_geojson)

    nl_land = world[world['ISO3166-1-Alpha-3'] == 'NLD'].copy()
    
    if nl_land.empty:
        print("❌ ОШИБКА: Нидерланды не найдены.")
        return

    print("3. Выбираю станции на западном побережье (возле Гааги/Роттердама)...")
    # ГЕОГРАФИЧЕСКИЙ ФИЛЬТР: Берем только станции с долготой < 4.65
    west_coast_stations = df_stations[df_stations['lon'] < 4.65].copy()
    print(f"   Найдено станций на западе: {len(west_coast_stations)}. Добавляю воду вокруг них...")

    st_geom = [Point(xy) for xy in zip(west_coast_stations['lon'], west_coast_stations['lat'])]
    gdf_west_stations = gpd.GeoDataFrame(west_coast_stations, geometry=st_geom, crs="EPSG:4326")

    # Переводим в метры (Web Mercator) для ровного буфера
    nl_land_metric = nl_land.to_crs(epsg=3857)
    west_stations_metric = gdf_west_stations.to_crs(epsg=3857)

    # Буфер ровно 20 км только для западного побережья
    west_buffers_metric = west_stations_metric.copy()
    west_buffers_metric['geometry'] = west_buffers_metric.geometry.buffer(20000)

    print("4. Склеиваю сушу и прибрежные воды Гааги в единый контур...")
    combined_mask_metric = pd.concat([nl_land_metric[['geometry']], west_buffers_metric[['geometry']]])
    combined_mask_metric = combined_mask_metric.dissolve()

    # Возврат в градусы
    combined_mask = combined_mask_metric.to_crs(epsg=4326)

    # ---------------------------------------------------------
    # ШАГ 3: ОБРЕЗКА И СОХРАНЕНИЕ
    # ---------------------------------------------------------
    print("5. Вырезаю финальную карту...")
    gdf_final = gpd.sjoin(gdf_grid, combined_mask, how="inner", predicate="within")

    gdf_final[['cell_lon', 'cell_lat', 'wind_speed']].to_csv(output_file, index=False)
    print(f"✅ Готово! Файл сохранен: {output_file}")

    # ---------------------------------------------------------
    # ШАГ 4: ВИЗУАЛИЗАЦИЯ
    # ---------------------------------------------------------
    print("Рисую результат...")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    combined_mask.boundary.plot(ax=ax, color='red', linewidth=1.5, label="Граница (с водой на западе)")
    
    scatter = ax.scatter(
        gdf_final['cell_lon'], gdf_final['cell_lat'], 
        c=gdf_final['wind_speed'], cmap='YlGnBu', s=15, alpha=0.9
    )
    
    # Рисуем все станции черным, а те, вокруг которых добавили воду - красным
    ax.scatter(df_stations['lon'], df_stations['lat'], color='black', s=10, label="Все станции")
    ax.scatter(west_coast_stations['lon'], west_coast_stations['lat'], color='red', s=15, label="Буферизированные станции")
    
    plt.colorbar(scatter, label='Средняя скорость ветра (м/с)')
    plt.title("Карта ветра (Точная суша + Оффшор у Гааги)")
    plt.xlabel("Долгота")
    plt.ylabel("Широта")
    plt.legend()
    
    plt.savefig("data/processed/wind_map_the_hague.png")
    plt.show()

if __name__ == "__main__":
    generate_perfect_wind_map()