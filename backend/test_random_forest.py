import unittest
import json
import pandas as pd
from pathlib import Path

class TestRandomForestComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).resolve().parent
        cls.data_dir = cls.base_dir / "data" / "processed"
        cls.reports_dir = cls.data_dir / "reports"

    def test_outputs_exist(self):
        """Verify all expected CSV and JSON files are generated."""
        expected_files = [
            self.data_dir / "random_forest_comparison_full.csv",
            self.data_dir / "random_forest_comparison_backend.csv",
            self.reports_dir / "rf_threshold_balance.csv",
            self.reports_dir / "rf_verification_metrics.json",
            self.reports_dir / "rf_verification_classification_report.csv",
            self.reports_dir / "rf_verification_confusion_matrix.csv",
            self.reports_dir / "rf_verification_feature_importance.csv",
            self.reports_dir / "rf_empirical_metrics.json",
            self.reports_dir / "rf_empirical_classification_report.csv",
            self.reports_dir / "rf_empirical_confusion_matrix.csv",
            self.reports_dir / "rf_empirical_feature_importance.csv"
        ]
        for filepath in expected_files:
            with self.subTest(filepath=filepath.name):
                self.assertTrue(filepath.exists(), f"File {filepath.name} does not exist!")

    def test_full_dataset_columns(self):
        """Verify the full comparison dataset contains original and newly generated model columns."""
        full_csv_path = self.data_dir / "random_forest_comparison_full.csv"
        df = pd.read_csv(full_csv_path)
        
        required_cols = [
            "rf_verification_prediction",
            "rf_verification_probability",
            "rf_empirical_target_has_turbine",
            "rf_empirical_prediction",
            "rf_empirical_probability",
            "rf_empirical_label"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Column '{col}' is missing in full CSV!")
            
        # Verify values are reasonable
        self.assertTrue(df["rf_empirical_target_has_turbine"].isin([0, 1]).all())
        self.assertTrue(df["rf_empirical_prediction"].isin([0, 1]).all())
        self.assertTrue(df["rf_empirical_label"].isin(["Empirically Suitable", "Empirically Unsuitable"]).all())
        self.assertTrue(((df["rf_verification_probability"] >= 0.0) & (df["rf_verification_probability"] <= 1.0)).all())
        self.assertTrue(((df["rf_empirical_probability"] >= 0.0) & (df["rf_empirical_probability"] <= 1.0)).all())

    def test_backend_dataset_columns(self):
        """Verify the backend comparison dataset contains only relevant optimized columns."""
        backend_csv_path = self.data_dir / "random_forest_comparison_backend.csv"
        df = pd.read_csv(backend_csv_path)
        
        expected_cols = [
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
        
        self.assertEqual(len(df.columns), len(expected_cols))
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Column '{col}' missing or named incorrectly in backend CSV!")

    def test_threshold_balance_report(self):
        """Verify that the threshold balance report was created correctly with 2000m selected."""
        threshold_path = self.reports_dir / "rf_threshold_balance.csv"
        df = pd.read_csv(threshold_path)
        
        required_cols = ["threshold_m", "positive_count", "negative_count", "positive_percentage"]
        for col in required_cols:
            self.assertIn(col, df.columns)
            
        # Check specific rows for the tested thresholds
        self.assertEqual(len(df), 4)
        self.assertTrue((df["threshold_m"] == [1000, 2000, 3000, 5000]).all())
        
        # Verify the positive percentage is inside 10-30% range for 2000m
        row_2000 = df[df["threshold_m"] == 2000].iloc[0]
        self.assertTrue(10.0 <= row_2000["positive_percentage"] <= 30.0)

    def test_metrics_jsons(self):
        """Verify that metrics JSON files are valid and contain essential validation keys."""
        verification_json = self.reports_dir / "rf_verification_metrics.json"
        empirical_json = self.reports_dir / "rf_empirical_metrics.json"
        
        with open(verification_json, "r") as f:
            v_metrics = json.load(f)
        with open(empirical_json, "r") as f:
            e_metrics = json.load(f)
            
        self.assertIn("accuracy", v_metrics)
        self.assertIn("f1_score_macro", v_metrics)
        self.assertIn("roc_auc_weighted_ovr", v_metrics)
        self.assertIn("accuracy", e_metrics)
        self.assertIn("f1_score_binary", e_metrics)
        self.assertIn("roc_auc", e_metrics)

        # Check values are in acceptable range (0 to 1)
        self.assertTrue(0.0 <= v_metrics["accuracy"] <= 1.0)
        self.assertTrue(0.0 <= e_metrics["accuracy"] <= 1.0)

    def test_feature_importances(self):
        """Verify that feature importances are computed and sorted correctly."""
        v_feat_path = self.reports_dir / "rf_verification_feature_importance.csv"
        e_feat_path = self.reports_dir / "rf_empirical_feature_importance.csv"
        
        v_feat = pd.read_csv(v_feat_path)
        e_feat = pd.read_csv(e_feat_path)
        
        required_cols = ["feature", "gini_importance", "permutation_importance_mean", "permutation_importance_std"]
        for col in required_cols:
            self.assertIn(col, v_feat.columns)
            self.assertIn(col, e_feat.columns)
            
        self.assertEqual(len(v_feat), 4) # wind_speed, population_density, dist_to_nearest_turbine_m, is_natura2000
        self.assertEqual(len(e_feat), 3) # wind_speed, population_density, is_natura2000
        
        # Verify they are sorted by gini_importance in descending order
        self.assertTrue(v_feat["gini_importance"].is_monotonic_decreasing)
        self.assertTrue(e_feat["gini_importance"].is_monotonic_decreasing)

if __name__ == "__main__":
    unittest.main()
