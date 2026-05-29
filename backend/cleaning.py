import pandas as pd
from pathlib import Path

def clean_wind_data():
    input_file = Path("data/processed/knmi_wind_daily_with_coords.csv")
    output_file = Path("data/processed/knmi_wind_clean.csv")
    
    if not input_file.exists():
        print("❌ File not found. Run the previous script first.")
        return

    print("Loading raw dataset...")
    df = pd.read_csv(input_file)
    initial_rows = len(df)
    
    print(f"Initial number of rows: {initial_rows}")
    print("\nStarting data cleaning...")

    # RULE 1: Date formatting
    # Transform 20240101 into a real date 2024-01-01
    df['date'] = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d')
    df = df.drop(columns=['YYYYMMDD']) # Remove old column

    # RULE 2: Remove missing values (NaN)
    # If there's no data for average wind or gusts, we don't need this row
    df_clean = df.dropna(subset=['FG', 'FHX', 'FXX', 'DDVEC']).copy()
    nan_dropped = initial_rows - len(df_clean)

    # RULE 3: Physical constraints (Sanity Checks)
    # 3.1 Wind cannot be negative
    df_clean = df_clean[df_clean['FG'] >= 0]
    
    # 3.2 Wind direction must be between 0 and 360 degrees
    df_clean = df_clean[(df_clean['DDVEC'] >= 0) & (df_clean['DDVEC'] <= 360)]
    
    # 3.3 Wind gust is always greater than or equal to average speed
    df_clean = df_clean[df_clean['FXX'] >= df_clean['FG']]
    
    # 3.4 Exclude unrealistic anomalies (e.g., speed > 60 m/s)
    # Historical maximum in the Netherlands was around 40-45 m/s.
    df_clean = df_clean[df_clean['FXX'] < 60]

    final_rows = len(df_clean)
    physics_dropped = (initial_rows - nan_dropped) - final_rows

    # COLUMN REORDERING (for convenience)
    cols = ['STN', 'station_name', 'lat', 'lon', 'date', 'DDVEC', 'FHVEC', 'FG', 'FHX', 'FXX']
    df_clean = df_clean[cols]

    # Save clean dataset
    df_clean.to_csv(output_file, index=False)

    # ---------------------------------------------------------
    # DATA QUALITY REPORT
    # ---------------------------------------------------------
    print("\n==================================================")
    print(" 🧹 DATA CLEANING REPORT")
    print("==================================================")
    print(f"Total rows before cleaning:    {initial_rows}")
    print(f"Removed due to empty values:   {nan_dropped} rows")
    print(f"Removed due to physics errors: {physics_dropped} rows")
    print("-" * 50)
    print(f"Total clean rows:              {final_rows}")
    print(f"Percentage of saved data:      {round((final_rows/initial_rows)*100, 2)}%")
    print("==================================================")
    print(f"✅ Clean file saved: {output_file}")

if __name__ == "__main__":
    clean_wind_data()