import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from pathlib import Path

def generate_perfect_wind_map():
    # Пути
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

    # Делаем сетку чуть шире крайних станций
    lon_min, lon_max = x_st.min() - 0.3, x_st.max() + 0.3
    lat_min, lat_max = y_st.min() - 0.3, y_st.max() + 0.3

    grid_lon = np.linspace(lon_min, lon_max, 200)
    grid_lat = np.linspace(lat_min, lat_max, 200)
    X_grid, Y_grid = np.meshgrid(grid_lon, grid_lat)

    # Запуск интерполяции
    rbf = Rbf(x_st, y_st, z_wind, function='inverse', power=2)
    Z_grid = rbf(X_grid, Y_grid)

    # Упаковка сетки для гео-библиотеки
    df_grid = pd.DataFrame({
        "cell_lon": X_grid.flatten(),
        "cell_lat": Y_grid.flatten(),
        "wind_speed": Z_grid.flatten()
    })
    grid_geom = [Point(xy) for xy in zip(df_grid['cell_lon'], df_grid['cell_lat'])]
    gdf_grid = gpd.GeoDataFrame(df_grid, geometry=grid_geom, crs="EPSG:4326")

    # ---------------------------------------------------------
    # ШАГ 2: СОЗДАНИЕ УМНОЙ МАСКИ (Суша + Оффшор)
    # ---------------------------------------------------------
    print("2. Загружаю карту мира и вырезаю Нидерланды...")
    url_geojson = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
    world = gpd.read_file(url_geojson)

    # Жестко используем правильную колонку, которую мы нашли
    nl_land = world[world['ISO3166-1-Alpha-3'] == 'NLD'].copy()
    
    if nl_land.empty:
        print("❌ КРИТИЧЕСКАЯ ОШИБКА: Нидерланды не найдены в файле GeoJSON.")
        return

    print("3. Ищу морские станции и строю 20-километровый буфер...")
    st_geom = [Point(xy) for xy in zip(x_st, y_st)]
    gdf_stations = gpd.GeoDataFrame(df_stations, geometry=st_geom, crs="EPSG:4326")

    # Разделяем станции на сухопутные и морские
    stations_on_land = gpd.sjoin(gdf_stations, nl_land, how="inner", predicate="within")
    sea_stations = gdf_stations[~gdf_stations['STN'].isin(stations_on_land['STN'])].copy()
    
    # Метрическая система для ровного круга в километрах (EPSG:3857)
    nl_land_metric = nl_land.to_crs(epsg=3857)
    sea_stations_metric = sea_stations.to_crs(epsg=3857)

    # Буфер ровно 20 000 метров для оффшорных зон
    sea_buffers_metric = sea_stations_metric.copy()
    sea_buffers_metric['geometry'] = sea_buffers_metric.geometry.buffer(20000)

    print("4. Склеиваю сушу и море в единый контур...")
    combined_mask_metric = pd.concat([nl_land_metric[['geometry']], sea_buffers_metric[['geometry']]])
    combined_mask_metric = combined_mask_metric.dissolve()

    # Возврат в градусы (EPSG:4326)
    combined_mask = combined_mask_metric.to_crs(epsg=4326)

    # ---------------------------------------------------------
    # ШАГ 3: ОБРЕЗКА И СОХРАНЕНИЕ
    # ---------------------------------------------------------
    print("5. Вырезаю финальную карту...")
    gdf_final = gpd.sjoin(gdf_grid, combined_mask, how="inner", predicate="within")

    # Сохраняем в CSV
    gdf_final[['cell_lon', 'cell_lat', 'wind_speed']].to_csv(output_file, index=False)
    print(f"✅ Идеальная сетка сохранена: {output_file}")
    print(f"📊 Итоговое количество клеток (Нидерланды + Оффшор): {len(gdf_final)}")

    # ---------------------------------------------------------
    # ШАГ 4: ВИЗУАЛИЗАЦИЯ
    # ---------------------------------------------------------
    print("Рисую результат (окно появится через пару секунд)...")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Красный контур - это наша умная маска
    combined_mask.boundary.plot(ax=ax, color='red', linewidth=1.5, label="Граница маски")
    
    # Цветная карта ветра
    scatter = ax.scatter(
        gdf_final['cell_lon'], gdf_final['cell_lat'], 
        c=gdf_final['wind_speed'], cmap='YlGnBu', s=15, alpha=0.9
    )
    
    # Точки станций для наглядности
    ax.scatter(x_st, y_st, color='black', s=10, label="Метеостанции")
    
    plt.colorbar(scatter, label='Средняя скорость ветра (м/с)')
    plt.title("Идеальная карта ветра (Суша + Оффшорные зоны)")
    plt.xlabel("Долгота")
    plt.ylabel("Широта")
    plt.legend()
    
    plt.savefig("data/processed/wind_map_perfect.png")
    plt.show()

if __name__ == "__main__":
    generate_perfect_wind_map()