from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score


# =========================================================
# Path handling
# =========================================================

def find_processed_dir() -> Path:
    """
    Supports running the script from different locations:
    - from backend/
    - from project root
    - directly by file path
    """

    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()

    candidates = [
        script_dir / "data" / "processed",          # if script is inside backend/
        cwd / "data" / "processed",                 # if running from backend/
        cwd / "backend" / "data" / "processed",     # if running from repo root
        script_dir.parent / "backend" / "data" / "processed",
    ]

    for candidate in candidates:
        if (candidate / "ml_dataset_final.csv").exists():
            return candidate
        if (candidate / "ml_dataset_finished.csv").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find data/processed containing ml_dataset_final.csv "
        "or ml_dataset_finished.csv."
    )


PROCESSED_DIR = find_processed_dir()
REPORT_DIR = PROCESSED_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CANDIDATES = [
    PROCESSED_DIR / "ml_dataset_final.csv",
    PROCESSED_DIR / "ml_dataset_finished.csv",
]

INPUT_CSV = next(path for path in INPUT_CANDIDATES if path.exists())

FULL_OUTPUT_CSV = PROCESSED_DIR / "ml_dataset_kmeans_full.csv"
BACKEND_OUTPUT_CSV = PROCESSED_DIR / "ml_dataset_kmeans_backend.csv"
CLUSTER_PROFILE_CSV = REPORT_DIR / "kmeans_cluster_profile.csv"
METRICS_JSON = REPORT_DIR / "kmeans_metrics.json"


# =========================================================
# Helpers
# =========================================================

def minmax(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    min_val = series.min()
    max_val = series.max()

    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series(0.5, index=series.index)

    return ((series - min_val) / (max_val - min_val)).clip(0, 1)


def inverse_minmax(series: pd.Series) -> pd.Series:
    """
    Converts a raw feature where lower is better into a score where higher is better.
    """
    return 1 - minmax(series)


def validate_input_columns(df: pd.DataFrame) -> None:
    required_columns = [
        "cell_lon",
        "cell_lat",
        "wind_speed",
        "is_natura2000",
        "dist_to_nearest_turbine_m",
        "population_density",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Input dataset is missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


# =========================================================
# Feature engineering
# =========================================================

def create_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates direction-corrected ML features.

    Higher value should always mean better suitability:
    - wind_score: higher wind speed is better
    - population_score: lower population density is better
    - infrastructure_score: smaller distance to existing turbines is treated as better
      because it may indicate nearby wind infrastructure / proven wind-development areas
    - natura_score: 1 means outside Natura 2000, 0 means inside Natura 2000
    """

    df = df.copy()

    # Clean numeric columns
    numeric_cols = [
        "cell_lon",
        "cell_lat",
        "wind_speed",
        "is_natura2000",
        "dist_to_nearest_turbine_m",
        "population_density",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)

    df = df.dropna(subset=numeric_cols).copy()

    after = len(df)

    if before != after:
        print(f"Dropped {before - after} rows with missing critical ML values.")

    df["is_natura2000"] = df["is_natura2000"].astype(int)

    # Main ML scores
    df["wind_score"] = minmax(df["wind_speed"])

    # Population is very skewed, so log transform prevents dense cities from dominating.
    df["population_score"] = inverse_minmax(
        np.log1p(df["population_density"])
    )

    # Distance to nearest turbine:
    # smaller distance = better proxy for infrastructure / existing feasible areas.
    df["infrastructure_score"] = inverse_minmax(
        np.log1p(df["dist_to_nearest_turbine_m"])
    )

    # Natura is a hard exclusion.
    df["natura_score"] = 1 - df["is_natura2000"]

    # A transparent rule-based score is useful for interpreting clusters.
    # Natura cells are set to 0 later.
    df["ml_suitability_score"] = (
        100
        * (
            0.60 * df["wind_score"]
            + 0.25 * df["population_score"]
            + 0.15 * df["infrastructure_score"]
        )
    ).round(2)

    # Hard exclusion for Natura 2000
    df.loc[df["is_natura2000"] == 1, "ml_suitability_score"] = 0

    return df


# =========================================================
# K-Means model
# =========================================================

def run_kmeans(df: pd.DataFrame, n_clusters: int = 3):
    """
    Runs K-Means only on non-Natura candidate cells.

    Natura 2000 cells are not clustered as candidates because they are excluded.
    """

    candidate_mask = df["is_natura2000"] == 0

    candidates = df[candidate_mask].copy()
    excluded = df[~candidate_mask].copy()

    print("\nCandidate cells outside Natura 2000:", len(candidates))
    print("Excluded Natura 2000 cells:", len(excluded))

    features = [
        "wind_score",
        "population_score",
        "infrastructure_score",
    ]

    if len(candidates) < n_clusters:
        raise ValueError(
            f"Not enough non-Natura rows for K-Means. "
            f"Rows: {len(candidates)}, clusters: {n_clusters}"
        )

    X = candidates[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=20,
    )

    candidates["kmeans_cluster"] = model.fit_predict(X_scaled)

    # Silhouette can be slow on large data, so use a sample.
    sample_size = min(5000, len(candidates))

    silhouette = silhouette_score(
        X_scaled,
        candidates["kmeans_cluster"],
        sample_size=sample_size,
        random_state=42,
    )

    davies_bouldin = davies_bouldin_score(
        X_scaled,
        candidates["kmeans_cluster"],
    )

    # Interpret clusters by their average suitability score.
    cluster_profile = (
        candidates
        .groupby("kmeans_cluster")
        .agg(
            cell_count=("kmeans_cluster", "count"),
            avg_ml_suitability_score=("ml_suitability_score", "mean"),
            avg_wind_speed=("wind_speed", "mean"),
            avg_wind_score=("wind_score", "mean"),
            avg_population_density=("population_density", "mean"),
            avg_population_score=("population_score", "mean"),
            avg_dist_to_nearest_turbine_m=("dist_to_nearest_turbine_m", "mean"),
            avg_infrastructure_score=("infrastructure_score", "mean"),
        )
        .reset_index()
        .sort_values("avg_ml_suitability_score", ascending=True)
        .reset_index(drop=True)
    )

    labels = [
        "Low ML suitability",
        "Medium ML suitability",
        "High ML suitability",
    ]

    cluster_label_map = {
        int(row["kmeans_cluster"]): labels[i]
        for i, row in cluster_profile.iterrows()
    }

    cluster_rank_map = {
        int(row["kmeans_cluster"]): i + 1
        for i, row in cluster_profile.iterrows()
    }

    candidates["kmeans_label"] = candidates["kmeans_cluster"].map(cluster_label_map)
    candidates["kmeans_rank"] = candidates["kmeans_cluster"].map(cluster_rank_map)

    # Boolean output for API/frontend
    candidates["ml_suitable"] = candidates["kmeans_label"] == "High ML suitability"

    # Natura excluded rows
    excluded["kmeans_cluster"] = -1
    excluded["kmeans_label"] = "Excluded: Natura 2000"
    excluded["kmeans_rank"] = 0
    excluded["ml_suitable"] = False

    # Combine back together
    output = pd.concat([candidates, excluded], ignore_index=True)

    # Keep same rough map order
    output = output.sort_values(["cell_lat", "cell_lon"]).reset_index(drop=True)

    # Add excluded row to the cluster profile for reporting
    excluded_profile = pd.DataFrame([{
        "kmeans_cluster": -1,
        "cell_count": len(excluded),
        "avg_ml_suitability_score": 0,
        "avg_wind_speed": excluded["wind_speed"].mean() if len(excluded) else np.nan,
        "avg_wind_score": excluded["wind_score"].mean() if len(excluded) else np.nan,
        "avg_population_density": excluded["population_density"].mean() if len(excluded) else np.nan,
        "avg_population_score": excluded["population_score"].mean() if len(excluded) else np.nan,
        "avg_dist_to_nearest_turbine_m": excluded["dist_to_nearest_turbine_m"].mean() if len(excluded) else np.nan,
        "avg_infrastructure_score": excluded["infrastructure_score"].mean() if len(excluded) else np.nan,
        "kmeans_label": "Excluded: Natura 2000",
        "kmeans_rank": 0,
    }])

    cluster_profile["kmeans_label"] = cluster_profile["kmeans_cluster"].map(cluster_label_map)
    cluster_profile["kmeans_rank"] = cluster_profile["kmeans_cluster"].map(cluster_rank_map)

    cluster_profile = pd.concat(
        [excluded_profile, cluster_profile],
        ignore_index=True,
    )

    metrics = {
        "model": "KMeans",
        "n_clusters": n_clusters,
        "rows_total": int(len(df)),
        "rows_clustered_non_natura": int(len(candidates)),
        "rows_excluded_natura2000": int(len(excluded)),
        "features_used_for_kmeans": features,
        "features_not_used_for_kmeans": [
            "cell_lon",
            "cell_lat",
            "is_natura2000"
        ],
        "reason_coordinates_not_used": (
            "Coordinates are kept for map visualisation but not used as ML features, "
            "because otherwise the model would cluster geography instead of suitability."
        ),
        "reason_natura_not_clustered": (
            "Natura 2000 is treated as a hard exclusion. Protected cells are labelled "
            "Excluded and are not treated as candidate turbine locations."
        ),
        "silhouette_score_sampled": float(silhouette),
        "silhouette_sample_size": int(sample_size),
        "davies_bouldin_score": float(davies_bouldin),
    }

    print("\nK-Means completed.")
    print(f"Silhouette score sampled: {silhouette:.4f}")
    print(f"Davies-Bouldin score: {davies_bouldin:.4f}")

    print("\nCluster profile:")
    print(cluster_profile.to_string(index=False))

    return output, cluster_profile, metrics


# =========================================================
# Save backend/frontend output
# =========================================================

def create_backend_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backend-ready dataset.

    The API can use this instead of ml_dataset_final.csv.
    """

    columns = [
        "cell_lon",
        "cell_lat",

        "wind_speed",
        "is_natura2000",
        "dist_to_nearest_turbine_m",
        "population_density",

        "wind_score",
        "population_score",
        "infrastructure_score",
        "natura_score",

        "ml_suitability_score",
        "ml_suitable",

        "kmeans_cluster",
        "kmeans_label",
        "kmeans_rank",
    ]

    return df[columns].copy()


# =========================================================
# Main
# =========================================================

def main():
    print("\nLoading ML dataset:")
    print(INPUT_CSV)

    df = pd.read_csv(INPUT_CSV)

    print("\nInput shape:")
    print(df.shape)

    print("\nInput columns:")
    print(df.columns.tolist())

    validate_input_columns(df)

    df = create_ml_features(df)

    output, cluster_profile, metrics = run_kmeans(df, n_clusters=3)

    backend_output = create_backend_output(output)

    output.to_csv(FULL_OUTPUT_CSV, index=False)
    backend_output.to_csv(BACKEND_OUTPUT_CSV, index=False)
    cluster_profile.to_csv(CLUSTER_PROFILE_CSV, index=False)

    with open(METRICS_JSON, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nSaved files:")
    print(f"Full ML output:     {FULL_OUTPUT_CSV}")
    print(f"Backend output:     {BACKEND_OUTPUT_CSV}")
    print(f"Cluster profile:    {CLUSTER_PROFILE_CSV}")
    print(f"Metrics:            {METRICS_JSON}")

    print("\nBackend output preview:")
    print(backend_output.head().to_string(index=False))


if __name__ == "__main__":
    main()