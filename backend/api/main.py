from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
from pathlib import Path
from typing import Optional

# --- APP INITIALIZATION ---
app = FastAPI(
    title="Wind Turbine Location API (Sprint 2)",
    description="Advanced location assessment considering Natura 2000, population density, and infrastructure.",
    version="2.0.0"
)

# Allow Next.js to communicate with our API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# --- HELPERS ---
def get_suitability_color(score: float, is_natura: int) -> str:
    """Returns a HEX color string based on suitability score (Blue -> Green style)."""
    if is_natura == 1:
        return "#64748b";  # Slate gray for Natura 2000
    
    # Blue -> Teal -> Green scale (Higher = More Green/Lighter)
    if score >= 80: return "#15803d" # Dark Green (Excellent)
    if score >= 70: return "#22c55e" # Green (Very Good)
    if score >= 60: return "#84cc16" # Lime (Good)
    if score >= 50: return "#14b8a6" # Teal (Average)
    if score >= 40: return "#0ea5e9" # Sky Blue (Moderate)
    if score >= 25: return "#2563eb" # Blue (Poor)
    return "#1e3a8a"                 # Dark Blue (Very Poor)

# --- GLOBAL VARIABLES (In-memory) ---
df_stations = None
df_grid = None
wind_kdtree = None
grid_coords = None

# --- LOAD DATA ON STARTUP ---
@app.on_event("startup")
async def load_data():
    global df_stations, df_grid, wind_kdtree, grid_coords
    
    base_dir = Path(__file__).parent.parent
    stations_path = base_dir / "data" / "processed" / "knmi_stations_summary.csv"
    
    # 🔴 LOADING NEW DATASET (with density and Natura 2000)
    grid_path = base_dir / "data" / "processed" / "ml_dataset_kmeans_full.csv"

    print("⏳ Loading data into server memory...")
    
    if stations_path.exists():
        df_stations = pd.read_csv(stations_path)
        print(f"✅ Successfully loaded {len(df_stations)} weather stations.")
    else:
        print("⚠️ WARNING: Stations file not found!")

    if grid_path.exists():
        df_grid = pd.read_csv(grid_path)
        
        # Override suitability based on the new 60% threshold
        df_grid['ml_suitable'] = df_grid['ml_suitability_score'] >= 60.0
        
        # Pre-calculate colors for the heatmap (avoids per-request overhead)
        df_grid['suitability_color'] = df_grid.apply(
            lambda row: get_suitability_color(row['ml_suitability_score'], int(row['is_natura2000'])),
            axis=1
        )

        grid_coords = df_grid[['cell_lat', 'cell_lon']].values
        wind_kdtree = cKDTree(grid_coords)
        print(f"✅ Successfully loaded {len(df_grid)} smart grid points with colors.")
    else:
        print("⚠️ WARNING: Grid file not found!")


# --- REQUEST SCHEMAS (Pydantic) ---
class EvaluateRequest(BaseModel):
    lat: float
    lon: float
    turbine_model: str = "Vestas_V164_8MW"


# ==========================================
# 1. GROUP: STATIONS
# ==========================================

@app.get("/api/v1/stations", tags=["Stations"])
async def get_all_stations():
    """Returns a list of all KNMI weather stations."""
    if df_stations is None:
        raise HTTPException(status_code=500, detail="Station data not loaded")
    return df_stations.to_dict(orient="records")

@app.get("/api/v1/stations/{station_id}", tags=["Stations"])
async def get_station_by_id(station_id: int):
    """Returns detailed information for a specific station (by STN)."""
    if df_stations is None:
        raise HTTPException(status_code=500, detail="Station data not loaded")
    
    station = df_stations[df_stations['STN'] == station_id]
    if station.empty:
        raise HTTPException(status_code=404, detail=f"Station with ID {station_id} not found")
    
    return station.iloc[0].to_dict()


# ==========================================
# 2. GROUP: WIND AND ENVIRONMENT (Environment Grid)
# ==========================================

@app.get("/api/v1/wind/point", tags=["Environment Data"])
async def get_environment_at_point(lat: float, lon: float):
    """Searches for the nearest cell in the ML dataset and returns all its characteristics."""
    if df_grid is None or wind_kdtree is None:
        raise HTTPException(status_code=500, detail="Grid not loaded")
    
    # Search for the nearest point (k=1)
    distance, index = wind_kdtree.query([lat, lon], k=1)
    
    # 0.08 degrees (approx. 8.8 km). 
    # This provides a balance: easy to click coastline points, 
    # but still restricts distant clicks from showing assessment popups.
    if distance > 0.08: 
        return {
            "requested_lat": lat,
            "requested_lon": lon,
            "error": "Location is too far from our data grid (Netherlands EEZ only)"
        }
    
    point = df_grid.iloc[index]
    
    return {
        "requested_lat": lat,
        "requested_lon": lon,
        "grid_lat": float(point["cell_lat"]),
        "grid_lon": float(point["cell_lon"]),

        "wind_speed_ms": round(float(point["wind_speed"]), 2),
        "is_natura2000": int(point["is_natura2000"]),
        "dist_to_nearest_turbine_m": int(point["dist_to_nearest_turbine_m"]),
        "population_density": int(point["population_density"]),

        "wind_score": round(float(point["wind_score"]), 3),
        "population_score": round(float(point["population_score"]), 3),
        "infrastructure_score": round(float(point["infrastructure_score"]), 3),
        "natura_score": round(float(point["natura_score"]), 3),

        "ml_suitability_score": round(float(point["ml_suitability_score"]), 2),
        "suitability_color": str(point["suitability_color"]),
        "ml_suitable": bool(point["ml_suitable"]),
        "kmeans_cluster": int(point["kmeans_cluster"]),
        "kmeans_label": str(point["kmeans_label"]),
        "kmeans_rank": int(point["kmeans_rank"]),

        "distance_deg": round(distance, 4)
    }

@app.get("/api/v1/wind/all", tags=["Environment Data"])
async def get_all_environment_data():
    """Returns the entire grid (for Heatmap rendering on the frontend)."""
    if df_grid is None:
        raise HTTPException(status_code=500, detail="Grid not loaded")
    
    # Return all 17,148 points at once
    return df_grid.to_dict(orient="records")


@app.get("/api/v1/wind/bbox", tags=["Environment Data"])
async def get_environment_bbox(
    min_lat: float = Query(..., description="Lower bound (South)"),
    max_lat: float = Query(..., description="Upper bound (North)"),
    min_lon: float = Query(..., description="Left bound (West)"),
    max_lon: float = Query(..., description="Right bound (East)")
):
    """Returns all grid points within a specified bounding box."""
    if df_grid is None:
        raise HTTPException(status_code=500, detail="Grid not loaded")
    
    mask = (
        (df_grid['cell_lat'] >= min_lat) & 
        (df_grid['cell_lat'] <= max_lat) & 
        (df_grid['cell_lon'] >= min_lon) & 
        (df_grid['cell_lon'] <= max_lon)
    )
    
    subset = df_grid[mask]
    return subset.to_dict(orient="records")


# ==========================================
# 3. GROUP: ZONES (Geometry)
# ==========================================

@app.get("/api/v1/zones/boundary", tags=["Zones"])
async def get_boundary_zone():
    """PLACEHOLDER: Returns GeoJSON of the Netherlands boundary + 30km EEZ."""
    return {
        "status": "not_implemented",
        "message": "In the future, GeoJSON with a red boundary will be returned here."
    }

@app.get("/api/v1/zones/exclusions", tags=["Zones"])
async def get_exclusion_zones():
    """PLACEHOLDER: Returns an array of restricted geo-zones (cities, parks)."""
    return {
        "status": "mock",
        "exclusions": []
    }


# ==========================================
# 4. GROUP: BUSINESS LOGIC EVALUATION (Evaluate)
# ==========================================

@app.post("/api/v1/turbines/evaluate", tags=["Evaluate"])
async def evaluate_location(request: EvaluateRequest):
    """
    ACTUAL BUSINESS LOGIC:
    Evaluates location suitability based on wind, parks, and population.
    """
    
    env = await get_environment_at_point(request.lat, request.lon)
    
    if "error" in env:
        return env # Return the error object directly (contains 'error' key)
    
    is_suitable = True
    score = 100
    warnings = []
    
    # 1. RULE: Natura 2000 (Strict Prohibition)
    if env["is_natura2000"] == 1:
        is_suitable = False
        score = 0
        warnings.append("❌ PROHIBITED: Natura 2000 protected area.")
        
    # 2. RULE: Wind (Economics)
    wind_speed = env["wind_speed_ms"]
    if wind_speed < 5.5:
        if is_suitable: is_suitable = False
        score -= 60
        warnings.append(f"💨 Low wind ({wind_speed} m/s). Project is not profitable.")
    elif wind_speed < 7.0:
        score -= 20
        warnings.append(f"💨 Moderate wind ({wind_speed} m/s). High masts required.")

    # 3. RULE: Population (Social Risk)
    pop_density = env["population_density"]
    if pop_density > 1000:
        if is_suitable: is_suitable = False
        score -= 40
        warnings.append(f"🏘️ High population density ({pop_density} people/km²). Risk of complaints.")
    elif pop_density > 300:
        score -= 15
        warnings.append(f"🏘️ Moderate population density ({pop_density} people/km²). Acoustic calculation required.")

    # 4. RULE: Infrastructure
    dist_turbine = env["dist_to_nearest_turbine_m"]
    if dist_turbine > 50000:
        score -= 10
        warnings.append(f"⚡ Isolated location ({(dist_turbine/1000):.1f} km to existing turbines). Expensive cabling.")

    score = max(0, score)

    return {
        "suitable": bool(env["ml_suitability_score"] >= 60.0), # Strict 60% threshold
        "score": env["ml_suitability_score"],
        "wind_speed_ms": env["wind_speed_ms"],
        "turbine_model": request.turbine_model,
        "ml_cluster": env["kmeans_cluster"],
        "ml_label": env["kmeans_label"],
        "warnings": warnings,
        "environment": env
    }
