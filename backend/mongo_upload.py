import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from pymongo import MongoClient, ASCENDING, DESCENDING
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "wind_turbine_suitability")
MONGO_GRID_COLLECTION = os.getenv("MONGO_GRID_COLLECTION", "grid_cells")
MONGO_STATIONS_COLLECTION = os.getenv("MONGO_STATIONS_COLLECTION", "knmi_stations")
MONGO_MODEL_RUNS_COLLECTION = os.getenv("MONGO_MODEL_RUNS_COLLECTION", "model_runs")

def get_suitability_color(score: float, is_natura: int) -> str:
    """Fallback color builder if missing."""
    if is_natura == 1:
        return "#64748b"
    if score >= 80: return "#15803d"
    if score >= 70: return "#22c55e"
    if score >= 60: return "#84cc16"
    if score >= 50: return "#14b8a6"
    if score >= 40: return "#0ea5e9"
    if score >= 25: return "#2563eb"
    return "#1e3a8a"

def main():
    print("🔌 Connecting to MongoDB Atlas/Local...")
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    
    grid_col = db[MONGO_GRID_COLLECTION]
    stations_col = db[MONGO_STATIONS_COLLECTION]
    runs_col = db[MONGO_MODEL_RUNS_COLLECTION]
    
    print(f"Using database: '{MONGO_DB}'")
    
    # ---------------------------------------------------------
    # 1. Clear Existing Collections for a clean load
    # ---------------------------------------------------------
    print("🧹 Cleaning up old database collections...")
    grid_col.delete_many({})
    stations_col.delete_many({})
    runs_col.delete_many({})
    
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data" / "processed"
    reports_dir = data_dir / "reports"
    
    # Determine the best comparison dataset to upload
    rf_path = data_dir / "random_forest_comparison_full.csv"
    kmeans_path = data_dir / "ml_dataset_kmeans_full.csv"
    
    input_path = None
    if rf_path.exists():
        input_path = rf_path
    elif kmeans_path.exists():
        input_path = kmeans_path
    else:
        raise FileNotFoundError("❌ Could not find any grid cell CSV files to upload.")
        
    print(f"📖 Loading grid cell data from: {input_path.name}")
    df = pd.read_csv(input_path)
    
    # Generate unique run ID
    model_run_id = f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"🏷️ Generated Model Run ID: {model_run_id}")
    
    # ---------------------------------------------------------
    # 2. Build Grid Documents & Bulk Insert
    # ---------------------------------------------------------
    print("🧱 Building grid cell documents...")
    grid_docs = []
    
    for idx, row in df.iterrows():
        lon = float(row["cell_lon"])
        lat = float(row["cell_lat"])
        
        # Features
        features = {
            "wind_speed": float(row["wind_speed"]),
            "is_natura2000": int(row["is_natura2000"]),
            "dist_to_nearest_turbine_m": int(row["dist_to_nearest_turbine_m"]),
            "population_density": int(row["population_density"])
        }
        
        # Scores
        scores = {
            "wind_score": float(row.get("wind_score", 0.0)),
            "population_score": float(row.get("population_score", 0.0)),
            "infrastructure_score": float(row.get("infrastructure_score", 0.0)),
            "natura_score": float(row.get("natura_score", 0.0)),
            "ml_suitability_score": float(row["ml_suitability_score"])
        }
        
        # KMeans
        kmeans = {
            "cluster": int(row["kmeans_cluster"]) if "kmeans_cluster" in row else None,
            "label": str(row["kmeans_label"]) if "kmeans_label" in row else None,
            "rank": int(row["kmeans_rank"]) if "kmeans_rank" in row else None,
            "suitable": bool(row["ml_suitable"]) if "ml_suitable" in row else False
        }
        
        # Random Forest with graceful checks
        random_forest = {
            "verification_prediction": str(row["rf_verification_prediction"]) if "rf_verification_prediction" in row and pd.notna(row["rf_verification_prediction"]) else None,
            "verification_probability": float(row["rf_verification_probability"]) if "rf_verification_probability" in row and pd.notna(row["rf_verification_probability"]) else None,
            "empirical_target_has_turbine": int(row["rf_empirical_target_has_turbine"]) if "rf_empirical_target_has_turbine" in row and pd.notna(row["rf_empirical_target_has_turbine"]) else None,
            "empirical_prediction": int(row["rf_empirical_prediction"]) if "rf_empirical_prediction" in row and pd.notna(row["rf_empirical_prediction"]) else None,
            "empirical_probability": float(row["rf_empirical_probability"]) if "rf_empirical_probability" in row and pd.notna(row["rf_empirical_probability"]) else None,
            "empirical_label": str(row["rf_empirical_label"]) if "rf_empirical_label" in row and pd.notna(row["rf_empirical_label"]) else None
        }
        
        # Display
        ml_suitable = bool(row["ml_suitable"]) if "ml_suitable" in row else (scores["ml_suitability_score"] >= 60.0)
        very_suitable = bool(scores["ml_suitability_score"] >= 70.0)
        suitability_color = str(row.get("suitability_color", get_suitability_color(scores["ml_suitability_score"], features["is_natura2000"])))
        
        display = {
            "ml_suitable": ml_suitable,
            "very_suitable": very_suitable,
            "suitability_color": suitability_color
        }
        
        doc = {
            "cell_id": f"cell_{lat:.6f}_{lon:.6f}",
            "cell_lon": lon,
            "cell_lat": lat,
            "location": {
                "type": "Point",
                "coordinates": [lon, lat] # Lon, Lat GeoJSON standard
            },
            "features": features,
            "scores": scores,
            "kmeans": kmeans,
            "random_forest": random_forest,
            "display": display,
            "model_run_id": model_run_id
        }
        grid_docs.append(doc)
        
    print(f"📤 Uploading {len(grid_docs)} grid cell documents in bulk...")
    grid_col.insert_many(grid_docs)
    print("✅ Successfully uploaded grid cells.")
    
    # ---------------------------------------------------------
    # 3. Load Stations Summary & Insert
    # ---------------------------------------------------------
    stations_path = data_dir / "knmi_stations_summary.csv"
    if stations_path.exists():
        print(f"📖 Loading KNMI weather stations from: {stations_path.name}")
        df_stations = pd.read_csv(stations_path)
        station_docs = []
        
        for idx, row in df_stations.iterrows():
            st_lon = float(row["lon"])
            st_lat = float(row["lat"])
            doc = {
                "station_id": int(row["STN"]),
                "station_name": str(row["station_name"]),
                "lat": st_lat,
                "lon": st_lon,
                "location": {
                    "type": "Point",
                    "coordinates": [st_lon, st_lat]
                },
                "metrics": {
                    "avg_wind_speed": float(row["avg_wind_speed"]) if pd.notna(row["avg_wind_speed"]) else 0.0,
                    "vector_wind_speed": float(row["vector_wind_speed"]) if pd.notna(row["vector_wind_speed"]) else 0.0,
                    "max_hourly_wind": float(row["max_hourly_wind"]) if pd.notna(row["max_hourly_wind"]) else 0.0,
                    "max_gust_speed": float(row["max_gust_speed"]) if pd.notna(row["max_gust_speed"]) else 0.0
                }
            }
            station_docs.append(doc)
            
        print(f"📤 Uploading {len(station_docs)} weather stations in bulk...")
        stations_col.insert_many(station_docs)
        print("✅ Successfully uploaded weather stations.")
    else:
        print("⚠️ Warning: knmi_stations_summary.csv not found. Skipping station upload.")
        
    # ---------------------------------------------------------
    # 4. Load Model Metrics & Metadata
    # ---------------------------------------------------------
    print("📈 Reading model metrics from reports...")
    metrics_run = {
        "run_id": model_run_id,
        "timestamp": pd.Timestamp.now().isoformat(),
        "metrics": {},
        "cluster_profile": [],
        "threshold_balance": []
    }
    
    # Load KMeans metrics
    km_metrics_path = reports_dir / "kmeans_metrics.json"
    if km_metrics_path.exists():
        with open(km_metrics_path, "r") as f:
            metrics_run["metrics"]["kmeans"] = json.load(f)
            
    # Load RF metrics
    rf_v_metrics_path = reports_dir / "rf_verification_metrics.json"
    if rf_v_metrics_path.exists():
        with open(rf_v_metrics_path, "r") as f:
            metrics_run["metrics"]["rf_verification"] = json.load(f)
            
    rf_e_metrics_path = reports_dir / "rf_empirical_metrics.json"
    if rf_e_metrics_path.exists():
        with open(rf_e_metrics_path, "r") as f:
            metrics_run["metrics"]["rf_empirical"] = json.load(f)
            
    # Load CSV profile reports
    cluster_profile_path = reports_dir / "kmeans_cluster_profile.csv"
    if cluster_profile_path.exists():
        metrics_run["cluster_profile"] = pd.read_csv(cluster_profile_path).to_dict(orient="records")
        
    threshold_balance_path = reports_dir / "rf_threshold_balance.csv"
    if threshold_balance_path.exists():
        metrics_run["threshold_balance"] = pd.read_csv(threshold_balance_path).to_dict(orient="records")
        
    print("📤 Inserting model run metadata document...")
    runs_col.insert_one(metrics_run)
    print("✅ Model run metadata registered.")

    # ---------------------------------------------------------
    # 5. Create Indexes
    # ---------------------------------------------------------
    print("\n⚡ Creating database indexes for performance...")
    
    # Grid cell indexes
    print("  - Creating index: 'location' 2dsphere on grid_cells")
    grid_col.create_index([("location", "2dsphere")])
    
    print("  - Creating index: (cell_lat, cell_lon) on grid_cells")
    grid_col.create_index([("cell_lat", ASCENDING), ("cell_lon", ASCENDING)])
    
    print("  - Creating index: 'scores.ml_suitability_score' DESC on grid_cells")
    grid_col.create_index([("scores.ml_suitability_score", DESCENDING)])
    
    print("  - Creating compound index: 'display.ml_suitable' + 'scores.ml_suitability_score' on grid_cells")
    grid_col.create_index([("display.ml_suitable", ASCENDING), ("scores.ml_suitability_score", DESCENDING)])
    
    print("  - Creating index: 'features.is_natura2000' on grid_cells")
    grid_col.create_index([("features.is_natura2000", ASCENDING)])
    
    print("  - Creating index: 'kmeans.label' on grid_cells")
    grid_col.create_index([("kmeans.label", ASCENDING)])
    
    # Station indexes
    print("  - Creating index: 'station_id' UNIQUE on knmi_stations")
    stations_col.create_index([("station_id", ASCENDING)], unique=True)
    
    print("  - Creating index: 'location' 2dsphere on knmi_stations")
    stations_col.create_index([("location", "2dsphere")])
    
    # Run metadata indexes
    print("  - Creating index: 'run_id' UNIQUE on model_runs")
    runs_col.create_index([("run_id", ASCENDING)], unique=True)
    
    print("🎉 Index creation complete!")

    # ---------------------------------------------------------
    # 6. Sample Records Output
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("📊 DATABASE UPLOAD COMPLETION REPORT")
    print("="*60)
    print(f"Total Grid Cells Uploaded      : {grid_col.count_documents({})}")
    print(f"Total KNMI Stations Uploaded   : {stations_col.count_documents({})}")
    print(f"Total Model Run Reports Saved  : {runs_col.count_documents({})}")
    
    print("\n👉 Example Grid Cell Document:")
    sample_cell = grid_col.find_one()
    if sample_cell:
        # Render a subset without BSON ObjectIds for clean print
        sample_cell.pop("_id", None)
        print(json.dumps(sample_cell, indent=2))
        
    print("\n👉 Example KNMI Station Document:")
    sample_station = stations_col.find_one()
    if sample_station:
        sample_station.pop("_id", None)
        print(json.dumps(sample_station, indent=2))
        
    print("\n👉 Latest Model Run Metadata ID:")
    sample_run = runs_col.find_one()
    if sample_run:
        print(f"  - Run ID   : {sample_run['run_id']}")
        print(f"  - Timestamp: {sample_run['timestamp']}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
