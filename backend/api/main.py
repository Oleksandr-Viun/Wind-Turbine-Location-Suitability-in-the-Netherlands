from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from scipy.spatial import cKDTree
from pathlib import Path
from typing import Optional

# --- ИНИЦИАЛИЗАЦИЯ ПРИЛОЖЕНИЯ ---
app = FastAPI(
    title="Wind Turbine Location API",
    description="API для оценки пригодности локаций под ветряки в Нидерландах и EEZ",
    version="1.0.0"
)

# Разрешаем Next.js общаться с нашим API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # В продакшене заменим на http://localhost:3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ (Оперативная память) ---
df_stations = None
df_grid = None
wind_kdtree = None
grid_coords = None

# --- ЗАГРУЗКА ДАННЫХ ПРИ СТАРТЕ ---
@app.on_event("startup")
async def load_data():
    global df_stations, df_grid, wind_kdtree, grid_coords
    
    base_dir = Path(__file__).parent.parent
    stations_path = base_dir / "data" / "processed" / "knmi_stations_summary.csv"
    grid_path = base_dir / "data" / "processed" / "wind_grid_final.csv"

    print("⏳ Загрузка данных в память сервера...")
    
    if stations_path.exists():
        df_stations = pd.read_csv(stations_path)
        print(f"✅ Успешно загружено {len(df_stations)} метеостанций.")
    else:
        print("⚠️ ВНИМАНИЕ: Файл станций не найден!")

    if grid_path.exists():
        df_grid = pd.read_csv(grid_path)
        grid_coords = df_grid[['cell_lat', 'cell_lon']].values
        wind_kdtree = cKDTree(grid_coords)
        print(f"✅ Успешно загружено {len(df_grid)} точек сетки ветра.")
    else:
        print("⚠️ ВНИМАНИЕ: Файл сетки ветра не найден!")


# --- СХЕМЫ ЗАПРОСОВ (Pydantic) ---
class EvaluateRequest(BaseModel):
    lat: float
    lon: float
    turbine_model: str = "Vestas_V164_8MW"


# ==========================================
# 1. ГРУППА: СТАНЦИИ (Stations)
# ==========================================

@app.get("/api/v1/stations", tags=["Stations"])
async def get_all_stations():
    """Возвращает список всех метеостанций KNMI."""
    if df_stations is None:
        raise HTTPException(status_code=500, detail="Данные станций не загружены")
    return df_stations.to_dict(orient="records")

@app.get("/api/v1/stations/{station_id}", tags=["Stations"])
async def get_station_by_id(station_id: int):
    """Возвращает детальную информацию по конкретной станции (по STN)."""
    if df_stations is None:
        raise HTTPException(status_code=500, detail="Данные станций не загружены")
    
    station = df_stations[df_stations['STN'] == station_id]
    if station.empty:
        raise HTTPException(status_code=404, detail=f"Станция с ID {station_id} не найдена")
    
    return station.iloc[0].to_dict()


# ==========================================
# 2. ГРУППА: ВЕТЕР (Wind Grid)
# ==========================================

@app.get("/api/v1/wind/point", tags=["Wind"])
async def get_wind_at_point(lat: float, lon: float):
    """Ищет скорость ветра в ближайшей точке нашей сгенерированной сетки."""
    if df_grid is None or wind_kdtree is None:
        raise HTTPException(status_code=500, detail="Сетка ветра не загружена")
    
    # Ищем ближайшую точку (k=1)
    distance, index = wind_kdtree.query([lat, lon], k=1)
    
    # 0.15 градуса (около 16 км). Если дальше - клик был вне нашей карты
    if distance > 0.15: 
        return {
            "requested_lat": lat,
            "requested_lon": lon,
            "error": "Локация находится слишком далеко от побережья Нидерландов или за пределами EEZ"
        }
    
    point_data = df_grid.iloc[index]
    
    return {
        "requested_lat": lat,
        "requested_lon": lon,
        "grid_lat": float(point_data['cell_lat']),
        "grid_lon": float(point_data['cell_lon']),
        "wind_speed_ms": round(float(point_data['wind_speed']), 2),
        "distance_deg": round(distance, 4)
    }

@app.get("/api/v1/wind/bbox", tags=["Wind"])
async def get_wind_bbox(
    min_lat: float = Query(..., description="Нижняя граница (Юг)"),
    max_lat: float = Query(..., description="Верхняя граница (Север)"),
    min_lon: float = Query(..., description="Левая граница (Запад)"),
    max_lon: float = Query(..., description="Правая граница (Восток)")
):
    """Возвращает все точки сетки внутри заданного прямоугольника (для отрисовки на фронтенде)."""
    if df_grid is None:
        raise HTTPException(status_code=500, detail="Сетка ветра не загружена")
    
    mask = (
        (df_grid['cell_lat'] >= min_lat) & 
        (df_grid['cell_lat'] <= max_lat) & 
        (df_grid['cell_lon'] >= min_lon) & 
        (df_grid['cell_lon'] <= max_lon)
    )
    
    subset = df_grid[mask]
    return subset.to_dict(orient="records")


# ==========================================
# 3. ГРУППА: ЗОНЫ (Геометрия)
# ==========================================

@app.get("/api/v1/zones/boundary", tags=["Zones"])
async def get_boundary_zone():
    """ЗАГЛУШКА: Возвращает GeoJSON границы Нидерландов + 30km EEZ."""
    return {
        "status": "not_implemented",
        "message": "В будущем здесь будет отдаваться GeoJSON с красной границей, чтобы фронтенд мог её нарисовать."
    }

@app.get("/api/v1/zones/exclusions", tags=["Zones"])
async def get_exclusion_zones():
    """ЗАГЛУШКА (Спринт 2): Возвращает массив запретных зон (города, парки)."""
    # Пока у нас нет датасета Natura 2000, отдаем пустой массив
    return {
        "status": "mock",
        "exclusions": []
    }


# ==========================================
# 4. ГРУППА: ОЦЕНКА БИЗНЕС-ЛОГИКИ (Evaluate)
# ==========================================

@app.post("/api/v1/turbines/evaluate", tags=["Evaluate"])
async def evaluate_location(request: EvaluateRequest):
    """
    Главный эндпоинт приложения: принимает координату и решает,
    можно ли там строить ветряк. (Текущая версия - Заглушка без запретных зон)
    """
    
    # 1. Получаем скорость ветра через наш другой эндпоинт
    wind_data = await get_wind_at_point(request.lat, request.lon)
    
    if "error" in wind_data:
        return {
            "suitable": False,
            "score": 0,
            "reason": wind_data["error"],
            "details": wind_data
        }
    
    wind_speed = wind_data["wind_speed_ms"]
    
    # 2. Мок-логика проверки пригодности (Score 0-100)
    is_suitable = True
    score = 100
    warnings = []
    
    # Простейшая логика рентабельности по ветру
    if wind_speed < 5.5:
        is_suitable = False
        score -= 60
        warnings.append("Слишком слабый ветер для экономической выгоды (< 5.5 м/с).")
    elif wind_speed < 7.0:
        score -= 20
        warnings.append("Средний ветер. Рекомендуется использовать турбины для слабого ветра с высокой башней.")
        
    return {
        "suitable": is_suitable,
        "score": score,
        "wind_speed_ms": wind_speed,
        "turbine_model": request.turbine_model,
        "warnings": warnings,
        "zones_checked": ["Mock Phase (Запретные зоны отключены)"]
    }