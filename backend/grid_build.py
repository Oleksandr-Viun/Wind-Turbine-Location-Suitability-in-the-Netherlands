import pandas as pd
import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from pathlib import Path

def generate_wind_grid_200x200():
    input_file = Path("data/processed/knmi_stations_summary.csv")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_file.exists():
        print("❌ Ошибка: Сначала нужно запустить скрипт слияния координат!")
        return

    # 1. Загружаем данные станций (там всего ~50 строк, по одной на станцию)
    df_stations = pd.read_csv(input_file)
    
    # Извлекаем координаты и целевую переменную (среднюю скорость ветра)
    x_stations = df_stations['lon'].values # Долгота (X)
    y_stations = df_stations['lat'].values # Широта (Y)
    z_wind = df_stations['avg_wind_speed'].values # Ветер (Z)

    print(f"Загружено {len(df_stations)} станций для построения сетки.")

    # 2. Определяем точные географические границы Нидерландов по нашим станциям
    lon_min, lon_max = x_stations.min() - 0.2, x_stations.max() + 0.2
    lat_min, lat_max = y_stations.min() - 0.2, y_stations.max() + 0.2

    # 3. ТЕХНИЧЕСКОЕ СОЗДАНИЕ СЕТКИ 200х200
    print("Создаю пустую координатную сетку 200x200...")
    grid_lon = np.linspace(lon_min, lon_max, 200)
    grid_lat = np.linspace(lat_min, lat_max, 200)
    
    # Meshgrid превращает векторы осей в полноценные двумерные матрицы координат
    X_grid, Y_grid = np.meshgrid(grid_lon, grid_lat)

    # 4. МАТЕМАТИЧЕСКАЯ ИНТЕРПОЛЯЦИЯ (Аналог IDW - функция Radial Basis Function)
    print("Запускаю алгоритм пространственной интерполяции ветра...")
    # 'inverse' как раз включает логику обратных расстояний (IDW)
    rbf_interpolator = Rbf(x_stations, y_stations, z_wind, function='inverse', power=2)
    
    # Рассчитываем значение ветра для каждой из 40 000 точек нашей сетки
    Z_wind_grid = rbf_interpolator(X_grid, Y_grid)

    # 5. СОХРАНЕНИЕ РЕЗУЛЬТАТА
    # Сохраняем сетку как матрицу NumPy (это стандарт в Data Science для хранения карт)
    np.save(output_dir / "wind_grid_200x200.npy", Z_wind_grid)
    
    # Также сохраним в плоскую таблицу CSV (40 000 строк), чтобы можно было загрузить в MongoDB
    grid_flat = pd.DataFrame({
        "cell_lon": X_grid.flatten(),
        "cell_lat": Y_grid.flatten(),
        "interpolated_wind": Z_wind_grid.flatten()
    })
    grid_flat.to_csv(output_dir / "wind_grid_flat.csv", index=False)
    print("✅ Сетка успешно рассчитана и сохранена в файлы!")

    # 6. ВИЗУАЛИЗАЦИЯ (Проверим, что получилось)
    print("Построение карты для визуальной проверки...")
    plt.figure(figsize=(10, 8))
    
    # Рисуем непрерывную цветную карту ветра
    contour = plt.pcolormesh(X_grid, Y_grid, Z_wind_grid, shading='auto', cmap='YlGnBu')
    plt.colorbar(contour, label='Средняя скорость ветра (м/с)')
    
    # Наносим сверху точки реальных метеостанций, чтобы видеть, откуда брались данные
    plt.scatter(x_stations, y_stations, color='red', marker='o', edgecolors='black', label='Станции KNMI')
    
    plt.title("Первая непрерывная карта ветра Нидерландов (Сетка 200x200)")
    plt.xlabel("Долгота (Longitude)")
    plt.ylabel("Широта (Latitude)")
    plt.legend()
    
    # Сохраняем картинку карты
    plt.savefig(output_dir / "wind_map_preview.png")
    print("✅ Карта-превью сохранена в data/processed/wind_map_preview.png")
    plt.show()

if __name__ == "__main__":
    generate_wind_grid_200x200()