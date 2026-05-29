import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon # <-- Добавили Polygon и MultiPolygon
from scipy.interpolate import Rbf
from pathlib import Path
import folium  
import webbrowser 
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# === ФУНКЦИЯ ДЛЯ ЗАЛИВКИ ВНУТРЕННИХ ДЫР ===
def fill_holes(geom):
    if isinstance(geom, Polygon):
        return Polygon(geom.exterior)
    elif isinstance(geom, MultiPolygon):
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])
    return geom
# ==========================================

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

    lon_min, lon_max = x_st.min() - 1.0, x_st.max() + 0.5
    lat_min, lat_max = y_st.min() - 0.5, y_st.max() + 1.0

    grid_lon = np.linspace(lon_min, lon_max, 250)
    grid_lat = np.linspace(lat_min, lat_max, 250)
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
    # ШАГ 2: УМНАЯ МАСКА (Суша + Прибрежное море 30 км без дыр)
    # ---------------------------------------------------------
    print("2. Загружаю карту суши...")
    url_land = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
    world = gpd.read_file(url_land)
    nl_land = world[world['ISO3166-1-Alpha-3'] == 'NLD'].copy()

    print("3. Скачиваю официальную морскую зону (EEZ)...")
    url_eez = "https://geo.vliz.be/geoserver/MarineRegions/ows?service=WFS&version=1.0.0&request=GetFeature&typeName=MarineRegions:eez&cql_filter=mrgid=5668&outputFormat=application/json"
    nl_eez = gpd.read_file(url_eez)

    print("4. GIS-магия: Заделываем внутренние дыры (Ваддензе и Зеландия)...")
    nl_land_metric = nl_land.to_crs(epsg=3857)
    nl_eez_metric = nl_eez.to_crs(epsg=3857)

    # Сливаем сушу и океан вместе (образуются внутренние дыры)
    combined_raw = pd.concat([nl_land_metric[['geometry']], nl_eez_metric[['geometry']]])
    combined_raw = combined_raw.dissolve()

    # Заливаем "бетоном" внутренние воды с помощью нашей функции
    combined_solid = combined_raw.copy()
    combined_solid['geometry'] = combined_solid.geometry.apply(fill_holes)

    print("5. Вырезаем ровно 30 км вдоль берега...")
    # Раздуваем сушу
    fat_land = nl_land_metric.copy()
    fat_land['geometry'] = fat_land.geometry.buffer(30000)

    # Оставляем только то, что попало в "монолитную" территорию Нидерландов
    final_mask_metric = gpd.overlay(fat_land, combined_solid, how='intersection')
    combined_mask = final_mask_metric.to_crs(epsg=4326)

    # ---------------------------------------------------------
    # ШАГ 3: ОБРЕЗКА И СОХРАНЕНИЕ
    # ---------------------------------------------------------
    print("6. Вырезаю финальную карту...")
    gdf_final = gpd.sjoin(gdf_grid, combined_mask, how="inner", predicate="within")

    gdf_final[['cell_lon', 'cell_lat', 'wind_speed']].to_csv(output_file, index=False)

    # ---------------------------------------------------------
    # ШАГ 4: ВИЗУАЛИЗАЦИЯ
    # ---------------------------------------------------------
    print("7. Генерирую интерактивную веб-карту...")
    
    m = gdf_final.explore(
        column="wind_speed",         
        cmap="YlGnBu",               
        tooltip="wind_speed",        
        marker_kwds={"radius": 4, "fill": True, "fillOpacity": 0.6}, 
        tiles="OpenStreetMap",       
        legend_kwds={"caption": "Средняя скорость ветра (м/с)"},
        name="Ветровая сетка"
    )

    combined_mask.boundary.explore(
        m=m,                         
        color="red",
        style_kwds={"weight": 2},    
        name="Граница зоны"
    )

    st_geom_all = [Point(xy) for xy in zip(df_stations['lon'], df_stations['lat'])]
    gdf_all_stations = gpd.GeoDataFrame(df_stations, geometry=st_geom_all, crs="EPSG:4326")
    
    gdf_all_stations.explore(
        m=m,
        color="black",
        marker_kwds={"radius": 5, "fill": True},
        tooltip=["STN", "station_name", "avg_wind_speed"], 
        name="Метеостанции KNMI"
    )

    folium.LayerControl().add_to(m)

    html_output = Path("data/processed/interactive_wind_map.html")
    m.save(html_output)
    
    print("✅ ГОТОВО! Открываю браузер...")
    
    file_path = os.path.abspath(html_output)
    webbrowser.open(f"file://{file_path}")

if __name__ == "__main__":
    generate_perfect_wind_map()