import pandas as pd
import numpy as np
from scipy.interpolate import NearestNDInterpolator
import matplotlib.pyplot as plt
from pathlib import Path
import math

def calculate_exact_regridding():
    input_file = Path("data/processed/knmi_stations_summary.csv")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_file.exists():
        print("❌ Ошибка: Сначала нужно сгенерировать summary станций!")
        return

    # 1. Загружаем данные станций
    df_stations = pd.read_csv(input_file)
    coords_stations = df_stations[['lon', 'lat']].values # (X, Y)
    wind_stations = df_stations['avg_wind_speed'].values # (Z)

    print(f"Загружено {len(df_stations)} станций.")

    # 2. Определяем границы сетки (чуть шире станций)
    lon_min, lon_max = coords_stations[:,0].min() - 0.2, coords_stations[:,0].max() + 0.2
    lat_min, lat_max = coords_stations[:,1].min() - 0.2, coords_stations[:,1].max() + 0.2

    # 3. Создаем пустую сетку 200x200
    print("Создаю координатную сетку 200x200...")
    grid_lon = np.linspace(lon_min, lon_max, 200)
    grid_lat = np.linspace(lat_min, lat_max, 200)
    X_grid, Y_grid = np.meshgrid(grid_lon, grid_lat)
    
    # 4. ИНТЕРПОЛЯЦИЯ БЛИЖАЙШЕГО СОСЕДА (Nearest Neighbor)
    # Это самый простой метод, чтобы показать точечный захват. 
    # Он не делает плавных переходов, но идеально держит границы.
    print("Запускаю алгоритм точного захвата (Nearest Neighbor)...")
    interpolator = NearestNDInterpolator(coords_stations, wind_stations)
    
    # Рассчитываем значение для всей сетки
    Z_wind_grid = interpolator(X_grid, Y_grid)

    # ==============================================================
    # 5. МАГИЯ ТОЧЕЧНОГО ОГРАНИЧЕНИЯ (Radius-based Exclusion)
    # Мы ставим NaN всем клеткам, которые находятся дальше 25 км от ЛЮБОЙ станции.
    # Это исключает морские клетки, где нет станций, и сухопутные клетки за границей.
    # ==============================================================
    print("Применяю радиусное ограничение 25 км...")
    Z_wind_grid_masked = Z_wind_grid.copy()
    
    # Расстояние в градусах (0.25 градуса ~ 25-30 км)
    R_LIMIT_DEG = 0.25 
    
    for i in range(200):
        for j in range(200):
            # Координаты текущей клетки
            c_lon, c_lat = X_grid[i, j], Y_grid[i, j]
            
            # Находим минимальное расстояние до любой из 50 станций
            min_dist = 999.0
            for s_lon, s_lat in coords_stations:
                # Простая теорема Пифагора (для малых расстояний это работает)
                dist = math.sqrt((c_lon - s_lon)**2 + (c_lat - s_lat)**2)
                if dist < min_dist:
                    min_dist = dist
            
            # Если клетка находится в пустыне (дальше 25 км от всех), мы её гасим
            if min_dist > R_LIMIT_DEG:
                Z_wind_grid_masked[i, j] = np.nan

    # 6. СОХРАНЕНИЕ
    # Сохраняем как плоскую таблицу
    grid_flat = pd.DataFrame({
        "cell_lon": X_grid.flatten(),
        "cell_lat": Y_grid.flatten(),
        "interpolated_wind": Z_wind_grid_masked.flatten()
    })
    
    # Удаляем пустые клетки (NaN) перед сохранением
    grid_flat_clean = grid_flat.dropna(subset=['interpolated_wind'])
    grid_flat_clean.to_csv(output_dir / "wind_grid_exact_masked.csv", index=False)
    print(f"✅ Сетка успешно сохранена в: {output_dir / 'wind_grid_exact_masked.csv'}")
    print(f"Захвачено клеток: {len(grid_flat_clean)} (из 40 000)")

    # 7. ВИЗУАЛИЗАЦИЯ
    print("\nРисую финальную точную карту...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Рисуем карту (pcolormesh умеет игнорировать NaN)
    contour = plt.pcolormesh(X_grid, Y_grid, Z_wind_grid_masked, shading='auto', cmap='YlGnBu')
    plt.colorbar(contour, label='Средняя скорость ветра (м/с)')
    
    # Наносим станции
    plt.scatter(coords_stations[:,0], coords_stations[:,1], color='red', marker='o', edgecolors='black', s=20)
    
    # Для проверки: нанесем контур Нидерландов (простой, по точкам)
    plt.title("Финальная точная карта ветра (Радиусный захват 25 км)")
    plt.xlabel("Долгота")
    plt.ylabel("Широта")
    
    plt.savefig(output_dir / "wind_map_exact_preview.png")
    plt.show()

if __name__ == "__main__":
    calculate_exact_regridding()