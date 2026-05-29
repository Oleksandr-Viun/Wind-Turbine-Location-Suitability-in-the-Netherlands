from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
import pandas as pd
from scipy.spatial import cKDTree
from pathlib import Path
from typing import Optional

# --- ИНИЦИАЛИЗАЦИЯ ПРИЛОЖЕНИЯ ---
app = FastAPI(
    title="Wind Turbine Location API (Sprint 2)",
    description="Продвинутая оценка локаций с учетом Natura 2000, плотности населения и инфраструктуры.",
    version="2.0.0"
)

# Разрешаем Next.js общаться с нашим API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
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
    
    # 🔴 ЗАГРУЖАЕМ НОВЫЙ ДАТАСЕТ (с плотностью и Natura 2000)
    grid_path = base_dir / "data" / "processed" / "ml_dataset_final.csv"

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
        print(f"✅ Успешно загружено {len(df_grid)} точек умной сетки.")
    else:
        print("⚠️ ВНИМАНИЕ: Файл сетки не найден!")


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
# 2. ГРУППА: ВЕТЕР И СРЕДА (Environment Grid)
# ==========================================

@app.get("/api/v1/wind/point", tags=["Environment Data"])
async def get_environment_at_point(lat: float, lon: float):
    """Ищет ближайшую клетку в ML-датасете и отдает ВСЕ ее характеристики."""
    if df_grid is None or wind_kdtree is None:
        raise HTTPException(status_code=500, detail="Сетка не загружена")
    
    # Ищем ближайшую точку (k=1)
    distance, index = wind_kdtree.query([lat, lon], k=1)
    
    # 0.15 градуса (около 16 км). Если дальше - клик был вне нашей карты
    if distance > 0.15: 
        return {
            "requested_lat": lat,
            "requested_lon": lon,
            "error": "Локация находится слишком далеко от побережья Нидерландов или за пределами EEZ"
        }
    
    point = df_grid.iloc[index]
    
    return {
        "requested_lat": lat,
        "requested_lon": lon,
        "grid_lat": float(point['cell_lat']),
        "grid_lon": float(point['cell_lon']),
        "wind_speed_ms": round(float(point['wind_speed']), 2),
        "is_natura2000": int(point['is_natura2000']),
        "dist_to_nearest_turbine_m": int(point['dist_to_nearest_turbine_m']),
        "population_density": int(point['population_density']),
        "distance_deg": round(distance, 4)
    }

@app.get("/api/v1/wind/all", tags=["Environment Data"])
async def get_all_environment_data():
    """Возвращает ВСЮ сетку целиком (для отрисовки Heatmap на фронтенде)."""
    if df_grid is None:
        raise HTTPException(status_code=500, detail="Сетка не загружена")
    
    # Отдаем все 17 148 точек разом
    return df_grid.to_dict(orient="records")


@app.get("/api/v1/wind/bbox", tags=["Environment Data"])
async def get_environment_bbox(
    min_lat: float = Query(..., description="Нижняя граница (Юг)"),
    max_lat: float = Query(..., description="Верхняя граница (Север)"),
    min_lon: float = Query(..., description="Левая граница (Запад)"),
    max_lon: float = Query(..., description="Правая граница (Восток)")
):
    """Возвращает все точки сетки внутри заданного прямоугольника."""
    if df_grid is None:
        raise HTTPException(status_code=500, detail="Сетка не загружена")
    
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
        "message": "В будущем здесь будет отдаваться GeoJSON с красной границей."
    }

@app.get("/api/v1/zones/exclusions", tags=["Zones"])
async def get_exclusion_zones():
    """ЗАГЛУШКА: Возвращает массив запретных гео-зон (города, парки)."""
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
    НАСТОЯЩАЯ БИЗНЕС-ЛОГИКА:
    Оценивает пригодность локации на основе ветра, парков и населения.
    """
    
    env = await get_environment_at_point(request.lat, request.lon)
    
    if "error" in env:
        return {
            "suitable": False,
            "score": 0,
            "reason": env["error"],
            "details": env
        }
    
    is_suitable = True
    score = 100
    warnings = []
    
    # 1. ПРАВИЛО: Natura 2000 (Строгий запрет)
    if env["is_natura2000"] == 1:
        is_suitable = False
        score = 0
        warnings.append("❌ ЗАПРЕЩЕНО: Территория природоохранной зоны Natura 2000.")
        
    # 2. ПРАВИЛО: Ветер (Экономика)
    wind_speed = env["wind_speed_ms"]
    if wind_speed < 5.5:
        if is_suitable: is_suitable = False
        score -= 60
        warnings.append(f"💨 Слабый ветер ({wind_speed} м/с). Проект нерентабелен.")
    elif wind_speed < 7.0:
        score -= 20
        warnings.append(f"💨 Средний ветер ({wind_speed} м/с). Нужны высокие мачты.")

    # 3. ПРАВИЛО: Население (Социальный риск)
    pop_density = env["population_density"]
    if pop_density > 1000:
        if is_suitable: is_suitable = False
        score -= 40
        warnings.append(f"🏘️ Высокая плотность населения ({pop_density} чел/км²). Риск жалоб.")
    elif pop_density > 300:
        score -= 15
        warnings.append(f"🏘️ Средняя плотность населения ({pop_density} чел/км²). Нужен акустический расчет.")

    # 4. ПРАВИЛО: Инфраструктура
    dist_turbine = env["dist_to_nearest_turbine_m"]
    if dist_turbine > 50000:
        score -= 10
        warnings.append(f"⚡ Изолированная локация ({(dist_turbine/1000):.1f} км до ветряков). Дорого тянуть кабель.")

    score = max(0, score)

    return {
        "suitable": is_suitable,
        "score": score,
        "wind_speed_ms": wind_speed,
        "turbine_model": request.turbine_model,
        "warnings": warnings,
        "environment": env
    }