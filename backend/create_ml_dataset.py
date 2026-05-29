import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
from scipy.spatial import cKDTree
import numpy as np
from pathlib import Path

def build_machine_learning_dataset():
    # Setup relative paths
    base_dir = Path("data/processed")
    raw_dir = Path("data/raw")
    
    print("1. Loading base wind grid (wind_grid_final.csv)...")
    df_grid = pd.read_csv(base_dir / "wind_grid_final.csv")
    grid_geom = [Point(xy) for xy in zip(df_grid['cell_lon'], df_grid['cell_lat'])]
    # Our grid is currently in degrees (EPSG:4326)
    gdf_grid = gpd.GeoDataFrame(df_grid, geometry=grid_geom, crs="EPSG:4326")
    
    # For accurate metric calculations, translate a copy of the grid to Dutch meters (RD New)
    gdf_grid_rd = gdf_grid.to_crs(epsg=28992)

    # =========================================================
    # FEATURE 1: NATURA 2000 RESTRICTIONS
    # =========================================================
    print("2. Processing Natura 2000 zones...")
    df_natura = pd.read_csv(raw_dir / "natura2000_pdok.csv")
    df_natura['geometry'] = df_natura['geometry_wkt'].apply(wkt.loads)
    # Natura 2000 is in degrees (EPSG:4326)
    gdf_natura = gpd.GeoDataFrame(df_natura, geometry='geometry', crs="EPSG:4326")
    
    # Point-in-polygon check
    joined_natura = gpd.sjoin(gdf_grid, gdf_natura, how="left", predicate="within")
    
    # If point is in a park, set to 1, otherwise 0
    df_grid['is_natura2000'] = (~joined_natura['index_right'].isna()).astype(int)
    print(f"   -> Found {df_grid['is_natura2000'].sum()} grid points inside nature reserves.")

    # =========================================================
    # FEATURE 2: EXISTING TURBINES (Distance in meters)
    # =========================================================
    print("3. Calculating distance to nearest existing turbines...")
    df_turbines = pd.read_csv(raw_dir / "windturbines_rivm_ashoogte.csv")
    df_turbines['geometry'] = df_turbines['geometry_wkt'].apply(wkt.loads)
    
    # IMPORTANT: Turbines were originally recorded in Dutch meters (EPSG:28992)
    gdf_turbines = gpd.GeoDataFrame(df_turbines, geometry='geometry', crs="EPSG:28992")
    
    # Extract X, Y (in meters) for both maps
    grid_coords_m = np.array(list(zip(gdf_grid_rd.geometry.x, gdf_grid_rd.geometry.y)))
    turbine_coords_m = np.array(list(zip(gdf_turbines.geometry.x, gdf_turbines.geometry.y)))
    
    # Use KD-Tree for instant nearest neighbor search
    tree = cKDTree(turbine_coords_m)
    distances, _ = tree.query(grid_coords_m, k=1)
    
    # Record distance in meters, rounded to integers
    df_grid['dist_to_nearest_turbine_m'] = np.round(distances).astype(int)

    # =========================================================
    # FEATURE 3: POPULATION DENSITY (Municipalities)
    # =========================================================
    print("4. Attaching population density from new .gpkg file...")
    # Load the merged file created in the previous step
    gdf_mun = gpd.read_file(base_dir / "municipality_full.gpkg")
    
    # Ensure CRS matches the grid (EPSG:4326)
    gdf_mun = gdf_mun.to_crs(epsg=4326)
    
    # Point-in-polygon check
    joined_mun = gpd.sjoin(gdf_grid, gdf_mun, how="left", predicate="within")
    
    # Extract density. If point is at sea (NaN), set to 0
    df_grid['population_density'] = joined_mun['population_density'].fillna(0).astype(int)

    # =========================================================
    # FINALIZATION AND SAVING
    # =========================================================
    print("5. Cleaning table and saving ML dataset...")
    # Keep only required columns, remove technical noise
    final_columns = [
        'cell_lon', 
        'cell_lat', 
        'wind_speed', 
        'is_natura2000', 
        'dist_to_nearest_turbine_m', 
        'population_density'
    ]
    df_ml = df_grid[final_columns]
    
    output_file = base_dir / "ml_dataset_final.csv"
    df_ml.to_csv(output_file, index=False)
    print(f"🎉 SUCCESS! Full dataset for model training saved: {output_file}")
    
    # Show preview of the first three rows
    print("\nPreview of your data:")
    print(df_ml.head(3).to_string())

if __name__ == "__main__":
    build_machine_learning_dataset()