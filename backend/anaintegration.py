import requests, json, time
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, shape
from shapely.ops import unary_union

OUTPUT_DIR  = Path("data/processed")
API_URL     = "https://archive-api.open-meteo.com/v1/archive"
GEO_URL     = ("https://raw.githubusercontent.com/datasets/"
               "geo-countries/master/data/countries.geojson")
START_DATE  = "2015-01-01"
END_DATE    = "2024-12-31"
SLEEP_S     = 0.6
VMIN, VMAX  = 2.0, 10.0
CMAP        = "YlGnBu"
COAST_M     = 25_000          # coastal buffer in metres

COUNTRIES = {
    "Denmark": {
        "geo_name":  "Denmark",
        "lat_range": (54.5, 57.9),
        "lon_range": (8.0,  15.3),
        "grid_lat":  8,
        "grid_lon":  10,
        "lat_clip":  None,
        "title":     "Denmark — Average Wind Speed 2015–2024 (ERA5)",
    },
    "Scotland": {
        "geo_name":  "United Kingdom",
        "lat_range": (54.5, 60.9),
        "lon_range": (-7.6, -0.7),
        "grid_lat":  9,
        "grid_lon":  7,
        "lat_clip":  54.5,          # clip UK to Scotland only
        "title":     "Scotland — Average Wind Speed 2015–2024 (ERA5)",
    },
    "France": {
        "geo_name":  "France",
        "lat_range": (42.3, 51.2),
        "lon_range": (-4.8,  8.3),
        "grid_lat":  9,
        "grid_lon":  11,
        "lat_clip":  None,
        "title":     "France — Average Wind Speed 2015–2024 (ERA5)",
    },
}

# ── Load world boundaries once ────────────────────────────────────────────────

def load_world() -> gpd.GeoDataFrame:
    print("Loading country boundaries...")
    r = requests.get(GEO_URL, timeout=30)
    r.raise_for_status()
    data = r.json()
    names = [f["properties"]["name"] for f in data["features"]]
    geoms = [shape(f["geometry"])     for f in data["features"]]
    world = gpd.GeoDataFrame({"name": names, "geometry": geoms}, crs="EPSG:4326")
    print(f"  Loaded {len(world)} countries.")
    return world


def get_boundary(world: gpd.GeoDataFrame,
                 geo_name: str,
                 lat_clip: float | None) -> gpd.GeoDataFrame:
    """
    Extract country polygon, optionally clip to lat >= lat_clip (for Scotland),
    then add a coastal buffer so nearby offshore cells are included.
    """
    country = world[world["name"] == geo_name].copy()
    if country.empty:
        print(f"  WARNING: '{geo_name}' not found.")
        return None

    country = country.to_crs("EPSG:4326")

    if lat_clip is not None:
        from shapely.geometry import box
        clip_box = box(-20, lat_clip, 5, 65)
        country = country.copy()
        country["geometry"] = country.geometry.intersection(clip_box)
        country = country[~country.geometry.is_empty]

    # Coastal buffer
    metric = country.to_crs("EPSG:3857")
    metric = metric.copy()
    metric["geometry"] = metric.geometry.buffer(COAST_M)
    return metric.to_crs("EPSG:4326")

# ── ERA5 data fetch ───────────────────────────────────────────────────────────

def fetch_point(lat: float, lon: float) -> float | None:
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": START_DATE, "end_date": END_DATE,
        "daily": "wind_speed_10m_mean",
        "wind_speed_unit": "ms", "timezone": "UTC",
    }
    try:
        r = requests.get(API_URL, params=params, timeout=30)
        r.raise_for_status()
        vals = r.json().get("daily", {}).get("wind_speed_10m_mean", [])
        valid = [v for v in vals if v is not None]
        return round(float(np.mean(valid)), 3) if valid else None
    except Exception as e:
        print(f"    Warning ({lat:.2f},{lon:.2f}): {e}")
        return None


def fetch_country_data(country: str, cfg: dict) -> pd.DataFrame:
    csv_path = OUTPUT_DIR / f"wind_stations_{country.lower()}.csv"

    # Use cached file if available
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"  {country}: loaded {len(df)} cached points from {csv_path.name}")
        return df

    lats = np.linspace(cfg["lat_range"][0], cfg["lat_range"][1], cfg["grid_lat"])
    lons = np.linspace(cfg["lon_range"][0], cfg["lon_range"][1], cfg["grid_lon"])
    grid = [(round(float(la), 3), round(float(lo), 3))
            for la in lats for lo in lons]
    total = len(grid)

    print(f"\n  {country}: fetching {total} ERA5 points "
          f"({cfg['grid_lat']}×{cfg['grid_lon']})...")
    rows = []
    for i, (lat, lon) in enumerate(grid, 1):
        speed = fetch_point(lat, lon)
        if speed is not None:
            print(f"    [{i:2d}/{total}] ({lat:.3f}, {lon:.3f}) → {speed:.2f} m/s")
            rows.append({"lat": lat, "lon": lon, "avg_wind_speed": speed})
        else:
            print(f"    [{i:2d}/{total}] ({lat:.3f}, {lon:.3f}) → FAILED")
        time.sleep(SLEEP_S)

    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"  Saved {len(df)} points → {csv_path}")
    return df

# ── RBF interpolation ─────────────────────────────────────────────────────────

def interpolate(df: pd.DataFrame, lat_range, lon_range, res=200):
    x = df["lon"].values
    y = df["lat"].values
    z = df["avg_wind_speed"].values
    grid_lon = np.linspace(lon_range[0], lon_range[1], res)
    grid_lat = np.linspace(lat_range[0], lat_range[1], res)
    X, Y = np.meshgrid(grid_lon, grid_lat)
    rbf = Rbf(x, y, z, function="inverse", power=2)
    Z   = np.clip(rbf(X, Y), 0, 25)
    return X, Y, Z

# ── Fast land mask via sjoin ──────────────────────────────────────────────────

def apply_mask(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
               boundary: gpd.GeoDataFrame) -> np.ndarray:
    """
    Mask grid cells outside the country boundary to NaN using
    a fast sjoin instead of a pixel-by-pixel loop.
    """
    if boundary is None:
        return Z

    rows, cols = Z.shape
    lats_flat  = Y.flatten()
    lons_flat  = X.flatten()

    gdf_pts = gpd.GeoDataFrame(
        {"row": np.repeat(np.arange(rows), cols),
         "col": np.tile(np.arange(cols), rows)},
        geometry=[Point(lo, la) for lo, la in zip(lons_flat, lats_flat)],
        crs="EPSG:4326",
    )

    inside = gpd.sjoin(gdf_pts, boundary[["geometry"]],
                       how="left", predicate="within")
    mask_flat = inside["index_right"].notna().values

    Z_masked = Z.copy().astype(float)
    Z_masked[~mask_flat.reshape(rows, cols)] = np.nan
    return Z_masked

# ── Draw map ──────────────────────────────────────────────────────────────────

def draw_map(country: str, cfg: dict,
             df: pd.DataFrame,
             X, Y, Z_masked,
             boundary: gpd.GeoDataFrame) -> None:

    fig, ax = plt.subplots(figsize=(9, 8))

    # Interpolated surface
    im = ax.pcolormesh(X, Y, Z_masked, shading="auto",
                       cmap=CMAP, vmin=VMIN, vmax=VMAX)

    # Country boundary (red outline, matching Netherlands style)
    if boundary is not None:
        boundary.boundary.plot(ax=ax, color="red", linewidth=1.3,
                               label="Boundary (incl. coastal zone)")

    # ERA5 sample point dots
    ax.scatter(df["lon"], df["lat"],
               c=df["avg_wind_speed"], cmap=CMAP,
               vmin=VMIN, vmax=VMAX,
               s=45, edgecolors="black", linewidths=0.7,
               zorder=5, label="Sample points (ERA5)")

    # Colorbar with 6 m/s threshold line
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Average wind speed (m/s)", fontsize=11)
    cbar.ax.axhline(y=6.0, color="red", linestyle="--", linewidth=1.5)
    cbar.ax.text(2.4, 6.05, "6 m/s\nthreshold",
                 color="red", fontsize=8, va="bottom")

    # Stats box
    w      = df["avg_wind_speed"]
    stats  = (f"Mean:    {w.mean():.2f} m/s\n"
              f"Max:     {w.max():.2f} m/s\n"
              f"Min:     {w.min():.2f} m/s\n"
              f"≥ 6 m/s: {(w >= 6).mean()*100:.0f}% of points")
    ax.text(0.03, 0.97, stats, transform=ax.transAxes,
            fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", alpha=0.78, linewidth=0.8))

    ax.set_title(cfg["title"], fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude",  fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.tick_params(labelsize=8)

    out = OUTPUT_DIR / f"wind_map_{country.lower()}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Map saved → {out}")
    plt.close()

# ── Comparison summary ────────────────────────────────────────────────────────

def print_summary(results: dict) -> None:
    print("\n" + "=" * 60)
    print("  WIND COMPARISON  |  ERA5 2015–2024  |  10 m height")
    print("=" * 60)
    print(f"  {'Country':<12}  {'Mean':>7}  {'Max':>7}  {'Min':>7}  {'≥6 m/s':>8}")
    print("-" * 60)
    for name, df in results.items():
        w = df["avg_wind_speed"]
        print(f"  {name:<12}  {w.mean():>7.2f}  {w.max():>7.2f}"
              f"  {w.min():>7.2f}  {(w>=6).mean()*100:>7.1f}%")
    print("=" * 60)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  WIND MAPS — Denmark, Scotland, France")
    print(f"  ERA5 via Open-Meteo  |  {START_DATE[:4]}–{END_DATE[:4]}")
    print("=" * 60)

    world   = load_world()
    results = {}

    for country, cfg in COUNTRIES.items():
        print(f"\n{'─'*55}\n  {country}\n{'─'*55}")

        # 1. Data
        df = fetch_country_data(country, cfg)
        if df.empty:
            print(f"  Skipping — no data.")
            continue
        results[country] = df

        # 2. Boundary
        boundary = get_boundary(world, cfg["geo_name"], cfg["lat_clip"])

        # 3. Interpolate
        print("  Interpolating 200×200 grid...")
        X, Y, Z = interpolate(df, cfg["lat_range"], cfg["lon_range"])

        # 4. Mask
        print("  Applying land mask (fast sjoin)...")
        Z_masked = apply_mask(X, Y, Z, boundary)
        n_inside = int(np.sum(~np.isnan(Z_masked)))
        print(f"  {n_inside} of {Z.size} grid cells inside boundary.")

        # 5. Draw
        print("  Drawing map...")
        draw_map(country, cfg, df, X, Y, Z_masked, boundary)

    if results:
        print_summary(results)


if __name__ == "__main__":
    main()