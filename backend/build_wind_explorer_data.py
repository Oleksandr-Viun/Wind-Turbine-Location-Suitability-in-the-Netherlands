import os
import time
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from scipy.interpolate import Rbf
from shapely.geometry import Point, shape
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "wind_turbine_suitability")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"

API_URL = "https://archive-api.open-meteo.com/v1/archive"
GEO_URL = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
SLEEP_S = 0.6
VMIN, VMAX = 2.0, 10.0
COAST_M = 25_000 # 25 km coastal buffer for offshore wind mapping

COUNTRIES = {
    "Denmark": {
        "geo_name": "Denmark",
        "lat_range": (54.5, 57.9),
        "lon_range": (8.0, 15.3),
        "grid_lat": 8,
        "grid_lon": 10,
        "lat_clip": None,
    },
    "Scotland": {
        "geo_name": "United Kingdom",
        "lat_range": (54.5, 60.9),
        "lon_range": (-7.6, -0.7),
        "grid_lat": 9,
        "grid_lon": 7,
        "lat_clip": 54.5,  # Clip UK to Scotland only
    },
    "France": {
        "geo_name": "France",
        "lat_range": (42.3, 51.2),
        "lon_range": (-4.8, 8.3),
        "grid_lat": 9,
        "grid_lon": 11,
        "lat_clip": None,
    },
    "Ireland": {
        "geo_name": "Ireland",
        "lat_range": (51.4, 55.5),
        "lon_range": (-10.7, -5.9),
        "grid_lat": 8,
        "grid_lon": 8,
        "lat_clip": None,
    }
}


# ==============================================================================
# PHASE 1: NETHERLANDS MONTHLY & ANNUAL COORDS INTERPOLATION
# ==============================================================================

def process_netherlands_monthly():
    print("\n🇳🇱 Processing Netherlands monthly & annual wind data...")
    
    # 1. Load exact grids
    grid_path = DATA_DIR / "ml_dataset_final.csv"
    if not grid_path.exists():
        print("❌ Error: ml_dataset_final.csv not found!")
        return None
    
    df_grid = pd.read_csv(grid_path)
    grid_coords = df_grid[["cell_lon", "cell_lat"]].drop_duplicates().values
    grid_lons = grid_coords[:, 0]
    grid_lats = grid_coords[:, 1]
    
    # 2. Load clean daily KNMI measurements
    clean_wind_path = DATA_DIR / "knmi_wind_clean.csv"
    if not clean_wind_path.exists():
        print("❌ Error: knmi_wind_clean.csv not found!")
        return None
        
    print("Reading KNMI clean measurements...")
    df_wind = pd.read_csv(clean_wind_path)
    df_wind["date"] = pd.to_datetime(df_wind["date"])
    df_wind["month"] = df_wind["date"].dt.month
    
    # Extract unique stations with their coords
    df_stations_geo = df_wind[["STN", "lat", "lon"]].drop_duplicates(subset=["STN"]).copy()
    
    # 3. Aggregate wind averages per station for annual and 1-12 months
    print("Aggregating wind speed averages per station...")
    # Annual
    df_annual = df_wind.groupby("STN")["FG"].mean().reset_index().rename(columns={"FG": "annual_speed"})
    df_stations_merged = pd.merge(df_stations_geo, df_annual, on="STN", how="inner")
    
    # Monthly (1-12)
    df_monthly = df_wind.groupby(["STN", "month"])["FG"].mean().reset_index()
    monthly_pivot = df_monthly.pivot(index="STN", columns="month", values="FG").reset_index()
    monthly_cols = {m: f"month_{m}" for m in range(1, 13)}
    monthly_pivot = monthly_pivot.rename(columns=monthly_cols)
    
    df_stations_merged = pd.merge(df_stations_merged, monthly_pivot, on="STN", how="inner")
    
    # 4. Interpolate periods onto the 17,148 grid points
    print("Spatially interpolating monthly & annual wind grids...")
    x_stations = df_stations_merged["lon"].values
    y_stations = df_stations_merged["lat"].values
    
    # Setup results DataFrame
    df_grid_interpolated = pd.DataFrame({
        "cell_lon": grid_lons,
        "cell_lat": grid_lats
    })
    
    # Interpolate Annual
    rbf_annual = Rbf(x_stations, y_stations, df_stations_merged["annual_speed"].values, function="inverse", power=2)
    df_grid_interpolated["annual_wind_speed"] = np.round(rbf_annual(grid_lons, grid_lats), 3)
    
    # Interpolate each month
    for m in range(1, 13):
        col_name = f"month_{m}"
        print(f"  Interpolating month {m}/12...")
        z_month = df_stations_merged[col_name].values
        rbf_month = Rbf(x_stations, y_stations, z_month, function="inverse", power=2)
        df_grid_interpolated[col_name] = np.round(rbf_month(grid_lons, grid_lats), 3)
        
    print(f"✅ Netherlands interpolation finished. Points count: {len(df_grid_interpolated)}")
    
    # Save a copy locally
    out_csv = DATA_DIR / "wind_explorer_netherlands_processed.csv"
    df_grid_interpolated.to_csv(out_csv, index=False)
    print(f"✅ Saved CSV fallback to {out_csv}")
    
    return df_grid_interpolated


# ==============================================================================
# PHASE 2: INTERNATIONAL ANNUAL DATA FETCHING, INTERPOLATION, AND MASKING
# ==============================================================================

def load_world() -> gpd.GeoDataFrame:
    print("Loading world boundaries...")
    try:
        r = requests.get(GEO_URL, timeout=30)
        r.raise_for_status()
        data = r.json()
        names = [f["properties"]["name"] for f in data["features"]]
        geoms = [shape(f["geometry"]) for f in data["features"]]
        world = gpd.GeoDataFrame({"name": names, "geometry": geoms}, crs="EPSG:4326")
        print(f"  Loaded {len(world)} country shapes.")
        return world
    except Exception as e:
        print(f"⚠️ Error loading country GeoJSON: {e}")
        return None

def get_boundary(world: gpd.GeoDataFrame, geo_name: str, lat_clip: float | None) -> gpd.GeoDataFrame:
    if world is None:
        return None
    country = world[world["name"] == geo_name].copy()
    if country.empty:
        print(f"  WARNING: Country '{geo_name}' not found in GeoJSON.")
        return None

    country = country.to_crs("EPSG:4326")

    if lat_clip is not None:
        from shapely.geometry import box
        clip_box = box(-20, lat_clip, 15, 65)
        country["geometry"] = country.geometry.intersection(clip_box)
        country = country[~country.geometry.is_empty]

    metric = country.to_crs("EPSG:3857")
    metric["geometry"] = metric.geometry.buffer(COAST_M)
    return metric.to_crs("EPSG:4326")

def fetch_point_era5(lat: float, lon: float) -> float | None:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": "wind_speed_10m_mean",
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }
    try:
        r = requests.get(API_URL, params=params, timeout=30)
        r.raise_for_status()
        vals = r.json().get("daily", {}).get("wind_speed_10m_mean", [])
        valid = [v for v in vals if v is not None]
        return round(float(np.mean(valid)), 3) if valid else None
    except Exception as e:
        print(f"    Warning ERA5 API fetch ({lat:.2f},{lon:.2f}): {e}")
        return None

def fetch_country_stations(country: str, cfg: dict) -> pd.DataFrame:
    csv_path = DATA_DIR / f"wind_stations_{country.lower()}.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"  {country}: Loaded {len(df)} cached sample points from {csv_path.name}")
        return df

    lats = np.linspace(cfg["lat_range"][0], cfg["lat_range"][1], cfg["grid_lat"])
    lons = np.linspace(cfg["lon_range"][0], cfg["lon_range"][1], cfg["grid_lon"])
    grid = [(round(float(la), 3), round(float(lo), 3)) for la in lats for lo in lons]
    total = len(grid)

    print(f"\n🌍 {country}: fetching {total} ERA5 points...")
    rows = []
    for i, (lat, lon) in enumerate(grid, 1):
        speed = fetch_point_era5(lat, lon)
        if speed is not None:
            print(f"    [{i:2d}/{total}] ({lat:.3f}, {lon:.3f}) → {speed:.2f} m/s")
            rows.append({"lat": lat, "lon": lon, "avg_wind_speed": speed})
        else:
            print(f"    [{i:2d}/{total}] ({lat:.3f}, {lon:.3f}) → FAILED")
        time.sleep(SLEEP_S)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"  Saved country points to {csv_path}")
    return df

def process_country_wind():
    print("\n🌍 Processing International wind grids...")
    world = load_world()
    
    country_records = []
    
    for country, cfg in COUNTRIES.items():
        print(f"\nProcessing {country}...")
        df_pts = fetch_country_stations(country, cfg)
        if df_pts.empty:
            print(f"⚠️ No sample points found for {country}, skipping.")
            continue
            
        boundary = get_boundary(world, cfg["geo_name"], cfg["lat_clip"])
        
        # Interpolate a 150x150 grid for country map resolution (balancing speed and detail)
        res = 150
        x = df_pts["lon"].values
        y = df_pts["lat"].values
        z = df_pts["avg_wind_speed"].values
        
        grid_lon = np.linspace(cfg["lon_range"][0], cfg["lon_range"][1], res)
        grid_lat = np.linspace(cfg["lat_range"][0], cfg["lat_range"][1], res)
        X, Y = np.meshgrid(grid_lon, grid_lat)
        
        rbf = Rbf(x, y, z, function="inverse", power=2)
        Z = np.clip(rbf(X, Y), 0, 25)
        
        # Land + Coastal mask
        if boundary is not None:
            print(f"  Applying boundary land mask for {country}...")
            rows, cols = Z.shape
            lats_flat = Y.flatten()
            lons_flat = X.flatten()
            
            gdf_pts = gpd.GeoDataFrame(
                {
                    "row": np.repeat(np.arange(rows), cols),
                    "col": np.tile(np.arange(cols), rows),
                    "val": Z.flatten()
                },
                geometry=[Point(lo, la) for lo, la in zip(lons_flat, lats_flat)],
                crs="EPSG:4326",
            )
            
            # Spatial join to mask out ocean cells beyond COAST_M
            inside = gpd.sjoin(gdf_pts, boundary[["geometry"]], how="left", predicate="within")
            mask_flat = inside["index_right"].notna().values
            
            # Extract valid points
            valid_idx = np.where(mask_flat)[0]
            print(f"  Selected {len(valid_idx)} grid cells inside boundary.")
            
            for idx in valid_idx:
                p = gdf_pts.iloc[idx]
                geom = p.geometry
                country_records.append({
                    "country": country,
                    "cell_lon": float(geom.x),
                    "cell_lat": float(geom.y),
                    "annual_wind_speed": float(round(p["val"], 3))
                })
        else:
            print(f"⚠️ No boundary shape for {country}, using flat box.")
            for r in range(res):
                for c in range(res):
                    country_records.append({
                        "country": country,
                        "cell_lon": float(X[r, c]),
                        "cell_lat": float(Y[r, c]),
                        "annual_wind_speed": float(round(Z[r, c], 3))
                    })
                    
    df_countries = pd.DataFrame(country_records)
    out_csv = DATA_DIR / "wind_explorer_countries_processed.csv"
    df_countries.to_csv(out_csv, index=False)
    print(f"🌍 Country processing finished. Total records: {len(df_countries)}")
    print(f"✅ Saved CSV fallback to {out_csv}")
    return df_countries


# ==============================================================================
# PHASE 3: UPLOAD TO MONGODB ATLAS & APPLY INDEXES
# ==============================================================================

def upload_to_mongodb(df_nl, df_countries):
    print("\n🔌 Uploading data to MongoDB Atlas...")
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        
        # 1. NETHERLANDS
        if df_nl is not None and not df_nl.empty:
            col_nl = db["wind_explorer_netherlands"]
            col_nl.drop()
            print("Uploading Netherlands wind explorer data...")
            
            nl_records = []
            for _, row in df_nl.iterrows():
                lon = float(row["cell_lon"])
                lat = float(row["cell_lat"])
                
                monthly = {str(m): float(row[f"month_{m}"]) for m in range(1, 13)}
                
                doc = {
                    "cell_id": f"cell_{lat:.6f}_{lon:.6f}",
                    "cell_lon": lon,
                    "cell_lat": lat,
                    "location": {
                        "type": "Point",
                        "coordinates": [lon, lat]
                    },
                    "annual_wind_speed": float(row["annual_wind_speed"]),
                    "monthly_wind_speed": monthly
                }
                nl_records.append(doc)
                
            col_nl.insert_many(nl_records)
            print(f"✅ Uploaded {len(nl_records)} Netherlands documents!")
            
            # Indexes
            print("Creating indexes on wind_explorer_netherlands...")
            col_nl.create_index([("location", "2dsphere")])
            col_nl.create_index([("cell_lat", 1), ("cell_lon", 1)])
            print("✅ Indexes created.")
            
        # 2. COUNTRIES
        if df_countries is not None and not df_countries.empty:
            col_countries = db["wind_explorer_countries"]
            col_countries.drop()
            print("Uploading Countries wind explorer data...")
            
            countries_records = []
            for _, row in df_countries.iterrows():
                lon = float(row["cell_lon"])
                lat = float(row["cell_lat"])
                
                doc = {
                    "country": str(row["country"]),
                    "cell_lon": lon,
                    "cell_lat": lat,
                    "location": {
                        "type": "Point",
                        "coordinates": [lon, lat]
                    },
                    "annual_wind_speed": float(row["annual_wind_speed"])
                }
                countries_records.append(doc)
                
            col_countries.insert_many(countries_records)
            print(f"✅ Uploaded {len(countries_records)} Country documents!")
            
            # Indexes
            print("Creating indexes on wind_explorer_countries...")
            col_countries.create_index([("location", "2dsphere")])
            col_countries.create_index([("cell_lat", 1), ("cell_lon", 1)])
            col_countries.create_index([("country", 1)])
            print("✅ Indexes created.")
            
    except Exception as e:
        print(f"❌ Failed to connect or upload to MongoDB Atlas: {e}")


def main():
    print("🚀 Starting Wind Explorer pipeline...")
    # 1. Process Netherlands
    df_nl = process_netherlands_monthly()
    
    # 2. Process Country comparison
    df_countries = process_country_wind()
    
    # 3. Upload and Index
    upload_to_mongodb(df_nl, df_countries)
    print("\n🎉 Wind Explorer Data Pipeline finished successfully!")

if __name__ == "__main__":
    main()
