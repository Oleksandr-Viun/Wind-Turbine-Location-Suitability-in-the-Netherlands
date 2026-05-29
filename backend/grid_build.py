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
        print("❌ Error: You must run the coordinate merging script first!")
        return

    # 1. Load station data (approx. 50 rows, one per station)
    df_stations = pd.read_csv(input_file)
    
    # Extract coordinates and target variable (average wind speed)
    x_stations = df_stations['lon'].values # Longitude (X)
    y_stations = df_stations['lat'].values # Latitude (Y)
    z_wind = df_stations['avg_wind_speed'].values # Wind (Z)

    print(f"Loaded {len(df_stations)} stations for grid construction.")

    # 2. Define exact geographical boundaries of the Netherlands based on our stations
    lon_min, lon_max = x_stations.min() - 0.2, x_stations.max() + 0.2
    lat_min, lat_max = y_stations.min() - 0.2, y_stations.max() + 0.2

    # 3. TECHNICAL CREATION OF 200x200 GRID
    print("Creating empty 200x200 coordinate grid...")
    grid_lon = np.linspace(lon_min, lon_max, 200)
    grid_lat = np.linspace(lat_min, lat_max, 200)
    
    # Meshgrid transforms axis vectors into full 2D coordinate matrices
    X_grid, Y_grid = np.meshgrid(grid_lon, grid_lat)

    # 4. MATHEMATICAL INTERPOLATION (IDW Analogue - Radial Basis Function)
    print("Starting spatial wind interpolation algorithm...")
    # 'inverse' includes inverse distance weighting (IDW) logic
    rbf_interpolator = Rbf(x_stations, y_stations, z_wind, function='inverse', power=2)
    
    # Calculate wind value for each of the 40,000 points in our grid
    Z_wind_grid = rbf_interpolator(X_grid, Y_grid)

    # 5. SAVING THE RESULT
    # Save the grid as a NumPy matrix (Data Science standard for storing maps)
    np.save(output_dir / "wind_grid_200x200.npy", Z_wind_grid)
    
    # Also save to a flat CSV table (40,000 rows) for MongoDB loading
    grid_flat = pd.DataFrame({
        "cell_lon": X_grid.flatten(),
        "cell_lat": Y_grid.flatten(),
        "interpolated_wind": Z_wind_grid.flatten()
    })
    grid_flat.to_csv(output_dir / "wind_grid_flat.csv", index=False)
    print("✅ Grid successfully calculated and saved to files!")

    # 6. VISUALIZATION (Verify the output)
    print("Building map for visual verification...")
    plt.figure(figsize=(10, 8))
    
    # Draw a continuous colored wind map
    contour = plt.pcolormesh(X_grid, Y_grid, Z_wind_grid, shading='auto', cmap='YlGnBu')
    plt.colorbar(contour, label='Average wind speed (m/s)')
    
    # Overlay real weather station points to see where the data came from
    plt.scatter(x_stations, y_stations, color='red', marker='o', edgecolors='black', label='KNMI Stations')
    
    plt.title("First Continuous Wind Map of the Netherlands (200x200 Grid)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    
    # Save map image
    plt.savefig(output_dir / "wind_map_preview.png")
    print("✅ Map preview saved to data/processed/wind_map_preview.png")
    plt.show()

if __name__ == "__main__":
    generate_wind_grid_200x200()