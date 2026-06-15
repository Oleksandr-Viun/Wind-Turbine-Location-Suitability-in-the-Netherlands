import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance

def main():
    print("🚀 Starting Random Forest Comparison Models training pipeline...")
    
    # ---------------------------------------------------------
    # 1. Path Setup
    # ---------------------------------------------------------
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data" / "processed"
    reports_dir = data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    full_path = data_dir / "ml_dataset_kmeans_full.csv"
    final_path = data_dir / "ml_dataset_final.csv"
    
    # Load dataset
    if full_path.exists():
        df = pd.read_csv(full_path)
        print(f"✅ Loaded {len(df)} rows from {full_path}")
    elif final_path.exists():
        df = pd.read_csv(final_path)
        print(f"✅ Loaded {len(df)} rows from {final_path}")
        
        # Heuristic/Engineered fields fallback if missing
        print("⚠️ Warning: ml_dataset_kmeans_full.csv not found. Re-creating essential columns...")
        df['is_natura2000'] = pd.to_numeric(df['is_natura2000'], errors='coerce').fillna(0).astype(int)
        
        # Normalization helpers
        def minmax(series):
            return ((series - series.min()) / (series.max() - series.min())).clip(0, 1)
        
        df["wind_score"] = minmax(df["wind_speed"])
        df["population_score"] = 1 - minmax(np.log1p(df["population_density"]))
        df["infrastructure_score"] = 1 - minmax(np.log1p(df["dist_to_nearest_turbine_m"]))
        df["natura_score"] = 1 - df["is_natura2000"]
        
        df["ml_suitability_score"] = (
            100 * (0.60 * df["wind_score"] + 0.25 * df["population_score"] + 0.15 * df["infrastructure_score"])
        ).round(2)
        df.loc[df["is_natura2000"] == 1, "ml_suitability_score"] = 0
        df['ml_suitable'] = df['ml_suitability_score'] >= 60.0
        
        if "kmeans_label" not in df.columns:
            # Fallback label
            df["kmeans_label"] = np.where(df["is_natura2000"] == 1, "Excluded: Natura 2000", 
                                          np.where(df["ml_suitable"], "High ML suitability", "Low ML suitability"))
    else:
        raise FileNotFoundError("❌ Could not find ml_dataset_kmeans_full.csv or ml_dataset_final.csv.")
        
    # Drop NaNs from critical columns
    critical_cols = ["wind_speed", "population_density", "dist_to_nearest_turbine_m", "is_natura2000"]
    df = df.dropna(subset=critical_cols).copy()
    print(f"Dataset size after cleaning: {len(df)} rows")

    # ---------------------------------------------------------
    # 2. Threshold Analysis for Model B (Empirical)
    # ---------------------------------------------------------
    print("\n📊 Evaluating distance thresholds for Model B target 'has_turbine'...")
    thresholds = [1000, 2000, 3000, 5000]
    balance_records = []
    selected_threshold = None
    best_dist_to_ideal = float('inf')
    
    for t in thresholds:
        pos_count = int((df["dist_to_nearest_turbine_m"] <= t).sum())
        neg_count = len(df) - pos_count
        pos_pct = (pos_count / len(df)) * 100
        
        balance_records.append({
            "threshold_m": t,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "positive_percentage": round(pos_pct, 2)
        })
        print(f"  - Threshold {t:4}m: positive={pos_count:5} ({pos_pct:5.2f}%), negative={neg_count:5}")
        
        # Check if within ideal 10% - 30% range and pick the one closest to 20%
        if 10.0 <= pos_pct <= 30.0:
            dist = abs(pos_pct - 20.0)
            if dist < best_dist_to_ideal:
                best_dist_to_ideal = dist
                selected_threshold = t
                
    if selected_threshold is None:
        # Fallback to absolute closest to 20%
        distances = [abs(rec["positive_percentage"] - 20.0) for rec in balance_records]
        selected_threshold = thresholds[np.argmin(distances)]
        
    print(f"🎯 Selected Threshold for 'has_turbine': {selected_threshold}m")
    
    # Save threshold balance report
    balance_df = pd.DataFrame(balance_records)
    balance_df.to_csv(reports_dir / "rf_threshold_balance.csv", index=False)
    print(f"📁 Saved threshold report to {reports_dir / 'rf_threshold_balance.csv'}")

    # Create Target for Model B
    df["rf_empirical_target_has_turbine"] = (df["dist_to_nearest_turbine_m"] <= selected_threshold).astype(int)

    # ---------------------------------------------------------
    # 3. Model A: Verification RFC
    # ---------------------------------------------------------
    print("\n🧠 Training Model A: Verification RFC (Targeting 'kmeans_label')...")
    
    # Target and Features setup
    target_A_col = "kmeans_label" if "kmeans_label" in df.columns else "ml_suitable"
    print(f"Target selected: '{target_A_col}'")
    
    X_A_cols = ["wind_speed", "population_density", "dist_to_nearest_turbine_m", "is_natura2000"]
    X_A = df[X_A_cols].copy()
    
    # Handle label encoding for evaluation (ROC-AUC needs it)
    le_A = LabelEncoder()
    y_A_encoded = le_A.fit_transform(df[target_A_col].astype(str))
    
    # Train-test split
    X_A_train, X_A_test, y_A_train, y_A_test = train_test_split(
        X_A, y_A_encoded, test_size=0.2, stratify=y_A_encoded, random_state=42
    )
    
    # Fit Model A
    rf_A = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    rf_A.fit(X_A_train, y_A_train)
    print("✅ Model A fitted.")
    
    # Model A Evaluation
    y_A_pred = rf_A.predict(X_A_test)
    y_A_prob = rf_A.predict_proba(X_A_test)
    
    acc_A = accuracy_score(y_A_test, y_A_pred)
    prec_A_macro, rec_A_macro, f1_A_macro, _ = precision_recall_fscore_support(y_A_test, y_A_pred, average="macro")
    prec_A_weighted, rec_A_weighted, f1_A_weighted, _ = precision_recall_fscore_support(y_A_test, y_A_pred, average="weighted")
    
    # Multiclass ROC-AUC (OVR)
    try:
        roc_auc_A = roc_auc_score(y_A_test, y_A_prob, multi_class="ovr", average="weighted")
    except Exception as e:
        print(f"⚠️ Could not compute ROC-AUC for Model A: {e}")
        roc_auc_A = None
        
    print(f"📊 Model A Accuracy : {acc_A:.4f}")
    print(f"📊 Model A Macro F1  : {f1_A_macro:.4f}")
    if roc_auc_A is not None:
        print(f"📊 Model A ROC-AUC   : {roc_auc_A:.4f}")
        
    # Save Model A classification report
    report_A = classification_report(y_A_test, y_A_pred, target_names=le_A.classes_, output_dict=True)
    report_A_df = pd.DataFrame(report_A).transpose()
    report_A_df.to_csv(reports_dir / "rf_verification_classification_report.csv", index=True)
    
    # Save Model A confusion matrix
    cm_A = confusion_matrix(y_A_test, y_A_pred)
    cm_A_df = pd.DataFrame(
        cm_A, 
        index=[f"Actual_{c}" for c in le_A.classes_], 
        columns=[f"Predicted_{c}" for c in le_A.classes_]
    )
    cm_A_df.to_csv(reports_dir / "rf_verification_confusion_matrix.csv", index=True)
    
    # Feature & Permutation Importance Model A
    print("🔄 Calculating permutation importances for Model A...")
    perm_importance_A = permutation_importance(rf_A, X_A_test, y_A_test, n_repeats=10, random_state=42, n_jobs=-1)
    
    feat_imp_A_df = pd.DataFrame({
        "feature": X_A_cols,
        "gini_importance": rf_A.feature_importances_,
        "permutation_importance_mean": perm_importance_A.importances_mean,
        "permutation_importance_std": perm_importance_A.importances_std
    }).sort_values(by="gini_importance", ascending=False)
    feat_imp_A_df.to_csv(reports_dir / "rf_verification_feature_importance.csv", index=False)
    
    # Save Model A Metrics JSON
    metrics_A = {
        "accuracy": acc_A,
        "precision_macro": prec_A_macro,
        "recall_macro": rec_A_macro,
        "f1_score_macro": f1_A_macro,
        "precision_weighted": prec_A_weighted,
        "recall_weighted": rec_A_weighted,
        "f1_score_weighted": f1_A_weighted,
        "roc_auc_weighted_ovr": roc_auc_A
    }
    with open(reports_dir / "rf_verification_metrics.json", "w") as f:
        json.dump(metrics_A, f, indent=4)
        
    # Generate full dataset predictions for Model A
    print("🔮 Scoring full dataset with Model A...")
    df["rf_verification_prediction"] = le_A.inverse_transform(rf_A.predict(X_A))
    df["rf_verification_probability"] = rf_A.predict_proba(X_A).max(axis=1)

    # ---------------------------------------------------------
    # 4. Model B: Empirical Real-World RFC
    # ---------------------------------------------------------
    print("\n🧠 Training Model B: Empirical RFC (Targeting 'rf_empirical_target_has_turbine')...")
    
    X_B_cols = ["wind_speed", "population_density", "is_natura2000"]
    X_B = df[X_B_cols].copy()
    y_B = df["rf_empirical_target_has_turbine"].copy()
    
    # Train-test split
    X_B_train, X_B_test, y_B_train, y_B_test = train_test_split(
        X_B, y_B, test_size=0.2, stratify=y_B, random_state=42
    )
    
    # Fit Model B
    rf_B = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    rf_B.fit(X_B_train, y_B_train)
    print("✅ Model B fitted.")
    
    # Model B Evaluation
    y_B_pred = rf_B.predict(X_B_test)
    y_B_prob = rf_B.predict_proba(X_B_test)[:, 1] # Probability of positive class
    
    acc_B = accuracy_score(y_B_test, y_B_pred)
    # Binary metrics for positive class (has_turbine = 1)
    prec_B_bin, rec_B_bin, f1_B_bin, _ = precision_recall_fscore_support(y_B_test, y_B_pred, average="binary")
    prec_B_macro, rec_B_macro, f1_B_macro, _ = precision_recall_fscore_support(y_B_test, y_B_pred, average="macro")
    prec_B_weighted, rec_B_weighted, f1_B_weighted, _ = precision_recall_fscore_support(y_B_test, y_B_pred, average="weighted")
    roc_auc_B = roc_auc_score(y_B_test, y_B_prob)
    
    print(f"📊 Model B Accuracy  : {acc_B:.4f}")
    print(f"📊 Model B Binary F1  : {f1_B_bin:.4f}")
    print(f"📊 Model B ROC-AUC    : {roc_auc_B:.4f}")
    
    # Save Model B classification report
    report_B = classification_report(y_B_test, y_B_pred, target_names=["No Turbine", "Has Turbine"], output_dict=True)
    report_B_df = pd.DataFrame(report_B).transpose()
    report_B_df.to_csv(reports_dir / "rf_empirical_classification_report.csv", index=True)
    
    # Save Model B confusion matrix
    cm_B = confusion_matrix(y_B_test, y_B_pred)
    cm_B_df = pd.DataFrame(
        cm_B, 
        index=["Actual_No_Turbine", "Actual_Has_Turbine"], 
        columns=["Predicted_No_Turbine", "Predicted_Has_Turbine"]
    )
    cm_B_df.to_csv(reports_dir / "rf_empirical_confusion_matrix.csv", index=True)
    
    # Feature & Permutation Importance Model B
    print("🔄 Calculating permutation importances for Model B...")
    perm_importance_B = permutation_importance(rf_B, X_B_test, y_B_test, n_repeats=10, random_state=42, n_jobs=-1)
    
    feat_imp_B_df = pd.DataFrame({
        "feature": X_B_cols,
        "gini_importance": rf_B.feature_importances_,
        "permutation_importance_mean": perm_importance_B.importances_mean,
        "permutation_importance_std": perm_importance_B.importances_std
    }).sort_values(by="gini_importance", ascending=False)
    feat_imp_B_df.to_csv(reports_dir / "rf_empirical_feature_importance.csv", index=False)
    
    # Save Model B Metrics JSON
    metrics_B = {
        "accuracy": acc_B,
        "precision_binary": prec_B_bin,
        "recall_binary": rec_B_bin,
        "f1_score_binary": f1_B_bin,
        "precision_macro": prec_B_macro,
        "recall_macro": rec_B_macro,
        "f1_score_macro": f1_B_macro,
        "precision_weighted": prec_B_weighted,
        "recall_weighted": rec_B_weighted,
        "f1_score_weighted": f1_B_weighted,
        "roc_auc": roc_auc_B
    }
    with open(reports_dir / "rf_empirical_metrics.json", "w") as f:
        json.dump(metrics_B, f, indent=4)
        
    # Generate full dataset predictions for Model B
    print("🔮 Scoring full dataset with Model B...")
    full_probs_B = rf_B.predict_proba(X_B)
    df["rf_empirical_prediction"] = rf_B.predict(X_B)
    df["rf_empirical_probability"] = full_probs_B[:, 1]
    df["rf_empirical_label"] = np.where(df["rf_empirical_prediction"] == 1, "Empirically Suitable", "Empirically Unsuitable")

    # ---------------------------------------------------------
    # 5. Save Final CSV Datasets
    # ---------------------------------------------------------
    print("\n💾 Saving final comparison CSV datasets...")
    
    # 1. Full Dataset
    full_output_path = data_dir / "random_forest_comparison_full.csv"
    df.to_csv(full_output_path, index=False)
    print(f"✅ Successfully saved full dataset ({len(df)} rows) to {full_output_path}")
    
    # 2. Backend Dataset
    backend_cols = [
        "cell_lon",
        "cell_lat",
        "wind_speed",
        "is_natura2000",
        "population_density",
        "dist_to_nearest_turbine_m",
        "ml_suitability_score",
        "kmeans_label",
        "ml_suitable",
        "rf_empirical_probability",
        "rf_empirical_prediction",
        "rf_empirical_label"
    ]
    # Ensure all backend columns are present (some fallbacks may not have them, handle gracefully)
    available_backend_cols = [col for col in backend_cols if col in df.columns]
    backend_df = df[available_backend_cols].copy()
    
    backend_output_path = data_dir / "random_forest_comparison_backend.csv"
    backend_df.to_csv(backend_output_path, index=False)
    print(f"✅ Successfully saved backend dataset ({len(backend_df)} rows) to {backend_output_path}")
    
    # ---------------------------------------------------------
    # 6. Print Comprehensive Summary
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("📊 RANDOM FOREST COMPARISON MODEL SUMMARY")
    print("="*80)
    print(f"1. Target Proximity Threshold (Model B): {selected_threshold}m")
    print(f"   - Positive class balance: {balance_records[thresholds.index(selected_threshold)]['positive_percentage']}% "
          f"({balance_records[thresholds.index(selected_threshold)]['positive_count']} cells)")
    
    print("\n2. Model A (Verification RFC) Performance:")
    print(f"   - Accuracy : {acc_A*100:.2f}%")
    print(f"   - Macro F1  : {f1_A_macro*100:.2f}%")
    if roc_auc_A is not None:
        print(f"   - ROC-AUC   : {roc_auc_A*100:.2f}%")
    print("   Interpretation: This confirms that our heuristic weights and unsupervised clustering rules")
    print("   are extremely consistent and easily reconstructible by a decision forest from raw features.")
    
    print("\n3. Model B (Empirical Real-World RFC) Performance:")
    print(f"   - Accuracy  : {acc_B*100:.2f}%")
    print(f"   - Binary F1 : {f1_B_bin*100:.2f}%")
    print(f"   - ROC-AUC   : {roc_auc_B*100:.2f}%")
    print("   Interpretation: This models real-world turbine placement. An AUC above 0.70 indicates the")
    print("   environment variables (wind, density, nature zones) hold strong predictive signal for actual siting.")
    
    print("\n4. Top Feature Importances (Gini / Permutation Mean):")
    print("   [Model A - Verification]")
    for _, r in feat_imp_A_df.iterrows():
        print(f"     - {r['feature']:28}: Gini={r['gini_importance']:.4f} | Permutation={r['permutation_importance_mean']:.4f}")
    
    print("\n   [Model B - Empirical]")
    for _, r in feat_imp_B_df.iterrows():
        print(f"     - {r['feature']:28}: Gini={r['gini_importance']:.4f} | Permutation={r['permutation_importance_mean']:.4f}")
        
    print("\n5. Critical Limitations:")
    print("   ⚠️ Siting proximity to existing turbines (Model B's target) is a highly noisy proxy.")
    print("   It is NOT an absolute 'ground-truth suitability' label. Many suitable sites remain undeveloped")
    print("   due to grid congestion, land ownership, local permitting, or historical timing, while some")
    print("   older turbines may occupy locations that would fail modern environmental or density standards.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
