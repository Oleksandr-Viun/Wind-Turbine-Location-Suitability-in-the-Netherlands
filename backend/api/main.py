import os
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from shapely.geometry import Point
from scipy.spatial import cKDTree
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "wind_turbine_suitability")
MONGO_GRID_COLLECTION = os.getenv("MONGO_GRID_COLLECTION", "grid_cells")
MONGO_STATIONS_COLLECTION = os.getenv("MONGO_STATIONS_COLLECTION", "knmi_stations")
MONGO_MODEL_RUNS_COLLECTION = os.getenv("MONGO_MODEL_RUNS_COLLECTION", "model_runs")
MONGO_EXPLORER_NL_COLLECTION = os.getenv("MONGO_EXPLORER_NL_COLLECTION", "wind_explorer_netherlands")
MONGO_EXPLORER_COUNTRIES_COLLECTION = os.getenv("MONGO_EXPLORER_COUNTRIES_COLLECTION", "wind_explorer_countries")

# --- APP INITIALIZATION ---
app = FastAPI(
    title="Wind Turbine Location API (Sprint 2 with MongoDB)",
    description="Advanced location assessment considering Natura 2000, population density, and infrastructure powered by MongoDB Atlas.",
    version="2.1.0"
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
db_client = None
df_explorer_nl = None
df_explorer_countries = None

# --- LOAD DATA ON STARTUP ---
@app.on_event("startup")
async def load_data():
    global df_stations, df_grid, wind_kdtree, grid_coords, db_client, df_explorer_nl, df_explorer_countries
    
    print("⏳ Connecting to MongoDB...")
    try:
        db_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Check connection
        db_client.server_info()
        db = db_client[MONGO_DB]
        print("✅ Connected to MongoDB successfully!")
        
        # 1. Load Stations from MongoDB knmi_stations collection
        print("⏳ Loading KNMI weather stations from MongoDB...")
        cursor_st = db[MONGO_STATIONS_COLLECTION].find()
        flat_st = []
        for doc in cursor_st:
            flat_st.append({
                "STN": doc["station_id"],
                "station_name": doc["station_name"],
                "lat": doc["lat"],
                "lon": doc["lon"],
                "avg_wind_speed": doc["metrics"]["avg_wind_speed"],
                "vector_wind_speed": doc["metrics"]["vector_wind_speed"],
                "max_hourly_wind": doc["metrics"]["max_hourly_wind"],
                "max_gust_speed": doc["metrics"]["max_gust_speed"]
            })
        df_stations = pd.DataFrame(flat_st)
        print(f"✅ Successfully loaded {len(df_stations)} weather stations from MongoDB.")
        
        # 2. Load Grid Cells from MongoDB grid_cells collection
        print("⏳ Loading smart grid cells from MongoDB...")
        cursor_grid = db[MONGO_GRID_COLLECTION].find()
        flat_grid = []
        for doc in cursor_grid:
            flat_grid.append({
                "cell_lon": doc["cell_lon"],
                "cell_lat": doc["cell_lat"],
                "wind_speed": doc["features"]["wind_speed"],
                "is_natura2000": doc["features"]["is_natura2000"],
                "dist_to_nearest_turbine_m": doc["features"]["dist_to_nearest_turbine_m"],
                "population_density": doc["features"]["population_density"],
                "wind_score": doc["scores"]["wind_score"],
                "population_score": doc["scores"]["population_score"],
                "infrastructure_score": doc["scores"]["infrastructure_score"],
                "natura_score": doc["scores"]["natura_score"],
                "ml_suitability_score": doc["scores"]["ml_suitability_score"],
                "kmeans_cluster": doc["kmeans"]["cluster"],
                "kmeans_label": doc["kmeans"]["label"],
                "kmeans_rank": doc["kmeans"]["rank"],
                "ml_suitable": doc["kmeans"]["suitable"],
                "suitability_color": doc["display"]["suitability_color"]
            })
        df_grid = pd.DataFrame(flat_grid)
        
        if len(df_grid) > 0:
            grid_coords = df_grid[['cell_lat', 'cell_lon']].values
            wind_kdtree = cKDTree(grid_coords)
            print(f"✅ Successfully loaded {len(df_grid)} smart grid points from MongoDB with cKDTree.")
        else:
            print("⚠️ WARNING: No grid points found in MongoDB! Trying filesystem fallback...")
            raise Exception("Grid cells collection is empty")
            
    except Exception as e:
        print(f"⚠️ ERROR connecting to MongoDB or loading data: {e}")
        print("🔄 Falling back to local file system...")
        
        base_dir = Path(__file__).resolve().parent.parent
        stations_path = base_dir / "data" / "processed" / "knmi_stations_summary.csv"
        
        # Determine the best comparison dataset as fallback
        rf_path = base_dir / "data" / "processed" / "random_forest_comparison_full.csv"
        kmeans_path = base_dir / "data" / "processed" / "ml_dataset_kmeans_full.csv"
        grid_path = rf_path if rf_path.exists() else kmeans_path
        
        if stations_path.exists():
            df_stations = pd.read_csv(stations_path)
            print(f"✅ Successfully loaded {len(df_stations)} weather stations (fallback).")
        else:
            print("⚠️ WARNING: Fallback stations file not found!")
            df_stations = pd.DataFrame()
            
        if grid_path.exists():
            df_grid = pd.read_csv(grid_path)
            df_grid['ml_suitable'] = df_grid['ml_suitability_score'] >= 60.0
            
            # Override/Calculate color if needed
            if 'suitability_color' not in df_grid.columns:
                df_grid['suitability_color'] = df_grid.apply(
                    lambda row: get_suitability_color(row['ml_suitability_score'], int(row['is_natura2000'])),
                    axis=1
                )
                
            grid_coords = df_grid[['cell_lat', 'cell_lon']].values
            wind_kdtree = cKDTree(grid_coords)
            print(f"✅ Successfully loaded {len(df_grid)} grid points from {grid_path.name} (fallback).")
        else:
            print("⚠️ WARNING: Fallback grid file not found!")
            df_grid = pd.DataFrame()

    # Load Wind Explorer caches unconditionally for fallback support
    base_dir = Path(__file__).resolve().parent.parent
    explorer_nl_path = base_dir / "data" / "processed" / "wind_explorer_netherlands_processed.csv"
    if explorer_nl_path.exists():
        df_explorer_nl = pd.read_csv(explorer_nl_path)
        print(f"✅ Loaded Netherlands wind explorer cache ({len(df_explorer_nl)} points).")
    else:
        print("⚠️ Netherlands wind explorer cache not found!")
        
    explorer_countries_path = base_dir / "data" / "processed" / "wind_explorer_countries_processed.csv"
    if explorer_countries_path.exists():
        df_explorer_countries = pd.read_csv(explorer_countries_path)
        print(f"✅ Loaded Countries wind explorer cache ({len(df_explorer_countries)} points).")
    else:
        print("⚠️ Countries wind explorer cache not found!")


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
    """Returns a list of all KNMI weather stations directly from MongoDB (or fallback in-memory)."""
    global df_stations
    if db_client is not None:
        try:
            cursor = db_client[MONGO_DB][MONGO_STATIONS_COLLECTION].find()
            stations = []
            for doc in cursor:
                stations.append({
                    "STN": doc["station_id"],
                    "station_name": doc["station_name"],
                    "lat": doc["lat"],
                    "lon": doc["lon"],
                    "avg_wind_speed": doc["metrics"]["avg_wind_speed"],
                    "vector_wind_speed": doc["metrics"]["vector_wind_speed"],
                    "max_hourly_wind": doc["metrics"]["max_hourly_wind"],
                    "max_gust_speed": doc["metrics"]["max_gust_speed"]
                })
            return stations
        except Exception as e:
            print(f"⚠️ MongoDB Stations query failed: {e}. Falling back to memory.")
            
    if df_stations is None or df_stations.empty:
        raise HTTPException(status_code=500, detail="Station data not loaded")
    return df_stations.to_dict(orient="records")

@app.get("/api/v1/stations/{station_id}", tags=["Stations"])
async def get_station_by_id(station_id: int):
    """Returns detailed information for a specific station (directly from MongoDB or fallback)."""
    global df_stations
    if db_client is not None:
        try:
            doc = db_client[MONGO_DB][MONGO_STATIONS_COLLECTION].find_one({"station_id": station_id})
            if doc:
                return {
                    "STN": doc["station_id"],
                    "station_name": doc["station_name"],
                    "lat": doc["lat"],
                    "lon": doc["lon"],
                    "avg_wind_speed": doc["metrics"]["avg_wind_speed"],
                    "vector_wind_speed": doc["metrics"]["vector_wind_speed"],
                    "max_hourly_wind": doc["metrics"]["max_hourly_wind"],
                    "max_gust_speed": doc["metrics"]["max_gust_speed"]
                }
        except Exception as e:
            print(f"⚠️ MongoDB Station by id failed: {e}. Falling back to memory.")

    if df_stations is None or df_stations.empty:
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
    """Searches for the nearest cell in the dataset and returns all its characteristics."""
    if df_grid is None or df_grid.empty or wind_kdtree is None:
        raise HTTPException(status_code=500, detail="Grid not loaded")
    
    # Search for the nearest point (k=1)
    distance, index = wind_kdtree.query([lat, lon], k=1)
    
    # 0.08 degrees (approx. 8.8 km)
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
        "kmeans_cluster": int(point["kmeans_cluster"]) if pd.notna(point["kmeans_cluster"]) else None,
        "kmeans_label": str(point["kmeans_label"]) if pd.notna(point["kmeans_label"]) else None,
        "kmeans_rank": int(point["kmeans_rank"]) if pd.notna(point["kmeans_rank"]) else None,

        "distance_deg": round(distance, 4)
    }

@app.get("/api/v1/wind/all", tags=["Environment Data"])
async def get_all_environment_data():
    """Returns the entire grid (for Heatmap rendering on the frontend)."""
    if df_grid is None or df_grid.empty:
        raise HTTPException(status_code=500, detail="Grid not loaded")
    
    return df_grid.to_dict(orient="records")


@app.get("/api/v1/wind/bbox", tags=["Environment Data"])
async def get_environment_bbox(
    min_lat: float = Query(..., description="Lower bound (South)"),
    max_lat: float = Query(..., description="Upper bound (North)"),
    min_lon: float = Query(..., description="Left bound (West)"),
    max_lon: float = Query(..., description="Right bound (East)")
):
    """Returns all grid points within a specified bounding box directly from MongoDB (or fallback memory)."""
    global df_grid
    if db_client is not None:
        try:
            query = {
                "cell_lat": {"$gte": min_lat, "$lte": max_lat},
                "cell_lon": {"$gte": min_lon, "$lte": max_lon}
            }
            cursor = db_client[MONGO_DB][MONGO_GRID_COLLECTION].find(query)
            subset = []
            for doc in cursor:
                subset.append({
                    "cell_lon": doc["cell_lon"],
                    "cell_lat": doc["cell_lat"],
                    "wind_speed": doc["features"]["wind_speed"],
                    "is_natura2000": doc["features"]["is_natura2000"],
                    "dist_to_nearest_turbine_m": doc["features"]["dist_to_nearest_turbine_m"],
                    "population_density": doc["features"]["population_density"],
                    "wind_score": doc["scores"]["wind_score"],
                    "population_score": doc["scores"]["population_score"],
                    "infrastructure_score": doc["scores"]["infrastructure_score"],
                    "natura_score": doc["scores"]["natura_score"],
                    "ml_suitability_score": doc["scores"]["ml_suitability_score"],
                    "kmeans_cluster": doc["kmeans"]["cluster"],
                    "kmeans_label": doc["kmeans"]["label"],
                    "kmeans_rank": doc["kmeans"]["rank"],
                    "ml_suitable": doc["kmeans"]["suitable"],
                    "suitability_color": doc["display"]["suitability_color"]
                })
            return subset
        except Exception as e:
            print(f"⚠️ MongoDB bounding box failed: {e}. Falling back to memory.")
            
    if df_grid is None or df_grid.empty:
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


# ==========================================
# 5. GROUP: MODEL METADATA & REPORTS
# ==========================================

@app.get("/api/v1/model-runs/latest", tags=["Model Runs"])
async def get_latest_model_run():
    """Returns the latest model run metadata directly from MongoDB."""
    if db_client is not None:
        try:
            doc = db_client[MONGO_DB][MONGO_MODEL_RUNS_COLLECTION].find_one(
                sort=[("timestamp", -1)]
            )
            if doc:
                doc.pop("_id", None)
                return doc
            else:
                raise HTTPException(status_code=404, detail="No model runs found in MongoDB")
        except HTTPException as he:
            raise he
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database error: {e}")
            
    raise HTTPException(status_code=501, detail="MongoDB is offline. Model metadata not available.")


# ==========================================
# 6. GROUP: WIND EXPLORER MODE
# ==========================================

@app.get("/api/v1/wind-explorer/netherlands", tags=["Wind Explorer"])
async def get_netherlands_wind_explorer(
    month: str = Query("annual", description="Selected month: 'annual', or '1'-'12'"),
    min_lat: Optional[float] = None,
    max_lat: Optional[float] = None,
    min_lon: Optional[float] = None,
    max_lon: Optional[float] = None
):
    """Returns grid coordinates with either annual or monthly average wind speeds for the Netherlands."""
    global df_explorer_nl, db_client
    
    # 1. Attempt MongoDB Query
    if db_client is not None:
        try:
            db = db_client[MONGO_DB]
            query = {}
            if min_lat is not None and max_lat is not None:
                query["cell_lat"] = {"$gte": min_lat, "$lte": max_lat}
            if min_lon is not None and max_lon is not None:
                query["cell_lon"] = {"$gte": min_lon, "$lte": max_lon}
                
            cursor = db[MONGO_EXPLORER_NL_COLLECTION].find(query)
            points = []
            for doc in cursor:
                # Select the correct wind speed based on selected month parameter
                if month == "annual":
                    speed = doc.get("annual_wind_speed", 0.0)
                else:
                    speed = doc.get("monthly_wind_speed", {}).get(month, 0.0)
                    
                points.append({
                    "cell_lon": doc["cell_lon"],
                    "cell_lat": doc["cell_lat"],
                    "wind_speed": round(speed, 3)
                })
            return points
        except Exception as e:
            print(f"⚠️ MongoDB Netherlands wind-explorer failed: {e}. Falling back to CSV.")
            
    # 2. Fallback to cached local Pandas dataframe
    if df_explorer_nl is None or df_explorer_nl.empty:
        raise HTTPException(status_code=500, detail="Netherlands Wind Explorer dataset not loaded")
        
    df_filtered = df_explorer_nl.copy()
    
    if min_lat is not None and max_lat is not None:
        df_filtered = df_filtered[(df_filtered["cell_lat"] >= min_lat) & (df_filtered["cell_lat"] <= max_lat)]
    if min_lon is not None and max_lon is not None:
        df_filtered = df_filtered[(df_filtered["cell_lon"] >= min_lon) & (df_filtered["cell_lon"] <= max_lon)]
        
    points = []
    col_name = "annual_wind_speed" if month == "annual" else f"month_{month}"
    
    if col_name not in df_filtered.columns:
        raise HTTPException(status_code=400, detail=f"Invalid month value: {month}")
        
    for _, row in df_filtered.iterrows():
        points.append({
            "cell_lon": float(row["cell_lon"]),
            "cell_lat": float(row["cell_lat"]),
            "wind_speed": round(float(row[col_name]), 3)
        })
        
    return points


@app.get("/api/v1/wind-explorer/country", tags=["Wind Explorer"])
async def get_country_wind_explorer(
    country: str = Query(..., description="Country name, e.g. 'Denmark', 'Scotland', 'France', 'Ireland'"),
    min_lat: Optional[float] = None,
    max_lat: Optional[float] = None,
    min_lon: Optional[float] = None,
    max_lon: Optional[float] = None
):
    """Returns grid coordinates with annual average wind speeds for the requested country."""
    global df_explorer_countries, db_client
    
    country_clean = country.strip().capitalize()
    
    if country_clean == "Netherlands":
        return await get_netherlands_wind_explorer(month="annual", min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon)
    
    # 1. Attempt MongoDB Query
    if db_client is not None:
        try:
            db = db_client[MONGO_DB]
            query = {"country": {"$regex": f"^{country_clean}$", "$options": "i"}}
            if min_lat is not None and max_lat is not None:
                query["cell_lat"] = {"$gte": min_lat, "$lte": max_lat}
            if min_lon is not None and max_lon is not None:
                query["cell_lon"] = {"$gte": min_lon, "$lte": max_lon}
                
            cursor = db[MONGO_EXPLORER_COUNTRIES_COLLECTION].find(query)
            points = []
            for doc in cursor:
                points.append({
                    "cell_lon": doc["cell_lon"],
                    "cell_lat": doc["cell_lat"],
                    "wind_speed": round(doc["annual_wind_speed"], 3)
                })
            return points
        except Exception as e:
            print(f"⚠️ MongoDB Country wind-explorer failed: {e}. Falling back to CSV.")
            
    # 2. Fallback to cached local Pandas dataframe
    if df_explorer_countries is None or df_explorer_countries.empty:
        raise HTTPException(status_code=500, detail="Countries Wind Explorer dataset not loaded")
        
    df_filtered = df_explorer_countries[df_explorer_countries["country"].str.lower() == country_clean.lower()].copy()
    
    if min_lat is not None and max_lat is not None:
        df_filtered = df_filtered[(df_filtered["cell_lat"] >= min_lat) & (df_filtered["cell_lat"] <= max_lat)]
    if min_lon is not None and max_lon is not None:
        df_filtered = df_filtered[(df_filtered["cell_lon"] >= min_lon) & (df_filtered["cell_lon"] <= max_lon)]
        
    points = []
    for _, row in df_filtered.iterrows():
        points.append({
            "cell_lon": float(row["cell_lon"]),
            "cell_lat": float(row["cell_lat"]),
            "wind_speed": round(float(row["annual_wind_speed"]), 3)
        })
        
    return points


@app.get("/api/v1/wind-explorer/point", tags=["Wind Explorer"])
async def get_wind_explorer_point_details(
    lat: float,
    lon: float,
    mode: str = Query("netherlands", description="'netherlands' or 'country'"),
    country: Optional[str] = None
):
    """Retrieves the wind speed of the grid cell nearest to the requested coordinates."""
    global df_explorer_nl, df_explorer_countries, db_client
    
    # 1. MongoDB $near spatial query
    if db_client is not None:
        try:
            db = db_client[MONGO_DB]
            collection_name = MONGO_EXPLORER_NL_COLLECTION if mode == "netherlands" else MONGO_EXPLORER_COUNTRIES_COLLECTION
            
            query = {}
            if mode == "country" and country:
                query["country"] = {"$regex": f"^{country.strip()}$", "$options": "i"}
                
            query["location"] = {
                "$near": {
                    "$geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat]
                    },
                    "$maxDistance": 50000 # 50km threshold
                }
            }
            
            doc = db[collection_name].find_one(query)
            if doc:
                if mode == "netherlands":
                    return {
                        "lat": doc["cell_lat"],
                        "lon": doc["cell_lon"],
                        "annual_wind_speed": doc["annual_wind_speed"],
                        "monthly_wind_speed": doc["monthly_wind_speed"]
                    }
                else:
                    return {
                        "lat": doc["cell_lat"],
                        "lon": doc["cell_lon"],
                        "country": doc["country"],
                        "annual_wind_speed": doc["annual_wind_speed"]
                    }
        except Exception as e:
            print(f"⚠️ MongoDB nearest spatial query failed: {e}. Falling back to distance math.")
            
    # 2. Local Pandas fallback KD-Tree/Math
    df_target = None
    if mode == "netherlands":
        df_target = df_explorer_nl
    else:
        if df_explorer_countries is not None and country:
            df_target = df_explorer_countries[df_explorer_countries["country"].str.lower() == country.strip().lower()]
        else:
            df_target = df_explorer_countries
            
    if df_target is None or df_target.empty:
        raise HTTPException(status_code=500, detail="Requested Wind Explorer Fallback dataset is empty")
        
    # Find nearest point using simple Euclidean distance (fine for small local grids)
    coords = df_target[["cell_lat", "cell_lon"]].values
    distances = np.sum((coords - np.array([lat, lon]))**2, axis=1)
    min_idx = np.argmin(distances)
    
    # Approx 50km check in degrees (0.5 degree)
    if distances[min_idx] > 0.25:
        raise HTTPException(status_code=404, detail="Coordinates are too far from the existing data grid")
        
    row = df_target.iloc[min_idx]
    
    if mode == "netherlands":
        monthly = {str(m): float(row[f"month_{m}"]) for m in range(1, 13)}
        return {
            "lat": float(row["cell_lat"]),
            "lon": float(row["cell_lon"]),
            "annual_wind_speed": float(row["annual_wind_speed"]),
            "monthly_wind_speed": monthly
        }
    else:
        return {
            "lat": float(row["cell_lat"]),
            "lon": float(row["cell_lon"]),
            "country": str(row["country"]),
            "annual_wind_speed": float(row["annual_wind_speed"])
        }
