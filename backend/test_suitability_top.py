import unittest
import asyncio
from api.main import (
    get_suitability_tiers,
    get_top_candidates,
    haversine_distance_m,
    load_data
)

class TestSuitabilityTopAndTiers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Trigger startup pre-loading of the fallback dataset and/or MongoDB connection
        asyncio.run(load_data())

    def test_suitability_tiers_function(self):
        """Verify that get_suitability_tiers returns correct schema and total cell count."""
        data = asyncio.run(get_suitability_tiers())
        
        self.assertIn("total_cells", data)
        self.assertIn("tiers", data)
        
        self.assertEqual(data["total_cells"], 17147)
        
        # Verify the tiers listed in the response
        tiers = {t["tier"]: t["count"] for t in data["tiers"]}
        expected_tiers = ["Excluded", "Low", "Medium", "Suitable", "Very Suitable"]
        for tier in expected_tiers:
            self.assertIn(tier, tiers)
            self.assertGreaterEqual(tiers[tier], 0)
            
        # Verify that total matches the sum of the tiers
        self.assertEqual(sum(tiers.values()), 17147)

    def test_suitability_top_function_default(self):
        """Verify that get_top_candidates returns default candidates sorting and schema."""
        data = asyncio.run(get_top_candidates())
        
        self.assertIn("scope", data)
        self.assertIn("limit", data)
        self.assertIn("min_score", data)
        self.assertIn("candidates", data)
        
        self.assertEqual(data["limit"], 20)
        self.assertEqual(data["min_score"], 60.0)
        
        candidates = data["candidates"]
        self.assertGreater(len(candidates), 0)
        self.assertLessEqual(len(candidates), 20)
        
        # Verify candidate values are sorted and exclude Natura 2000
        prev_score = 101.0
        for i, cand in enumerate(candidates, 1):
            self.assertEqual(cand["rank"], i)
            self.assertEqual(cand["is_natura2000"], 0)
            self.assertGreaterEqual(cand["ml_suitability_score"], 60.0)
            self.assertLessEqual(cand["ml_suitability_score"], prev_score)
            prev_score = cand["ml_suitability_score"]
            
            # Check fields
            self.assertIn("cell_lat", cand)
            self.assertIn("cell_lon", cand)
            self.assertIn("short_reason", cand)
            self.assertIn("kmeans_label", cand)

    def test_suitability_top_min_score_filter(self):
        """Verify that get_top_candidates respects min_score filter."""
        data = asyncio.run(get_top_candidates(min_score=80.0, limit=5))
        self.assertEqual(data["min_score"], 80.0)
        
        candidates = data["candidates"]
        for cand in candidates:
            self.assertGreaterEqual(cand["ml_suitability_score"], 80.0)

    def test_suitability_top_bbox_filter(self):
        """Verify that get_top_candidates filters geographically by bounding box."""
        # Query candidates for a box around The Hague / Rotterdam
        data = asyncio.run(get_top_candidates(
            min_lat=51.9,
            max_lat=52.1,
            min_lon=4.2,
            max_lon=4.5,
            limit=10
        ))
        self.assertEqual(data["scope"], "bbox")
        
        candidates = data["candidates"]
        for cand in candidates:
            self.assertTrue(51.9 <= cand["cell_lat"] <= 52.1)
            self.assertTrue(4.2 <= cand["cell_lon"] <= 4.5)

    def test_suitability_top_diversity_filter(self):
        """Verify that diversity filter avoids clustered candidates within min_distance_m."""
        # Diverse selection
        data = asyncio.run(get_top_candidates(diverse=True, min_distance_m=2000.0, limit=10))
        candidates_div = data["candidates"]
        
        # Check that no two points in the diverse set are within 2000m of each other
        for i in range(len(candidates_div)):
            for j in range(i + 1, len(candidates_div)):
                c1 = candidates_div[i]
                c2 = candidates_div[j]
                dist = haversine_distance_m(c1["cell_lat"], c1["cell_lon"], c2["cell_lat"], c2["cell_lon"])
                self.assertGreaterEqual(dist, 2000.0, f"Points {i} and {j} are too close: {dist:.1f}m")

    def test_suitability_top_non_diverse_clumping(self):
        """Verify that diverse=false lets candidates clump near each other."""
        data = asyncio.run(get_top_candidates(diverse=False, limit=5))
        candidates_clump = data["candidates"]
        
        # Typically, without diversity, top candidates will clump very close to each other
        has_clump = False
        for i in range(len(candidates_clump)):
            for j in range(i + 1, len(candidates_clump)):
                c1 = candidates_clump[i]
                c2 = candidates_clump[j]
                dist = haversine_distance_m(c1["cell_lat"], c1["cell_lon"], c2["cell_lat"], c2["cell_lon"])
                if dist < 1500.0:
                    has_clump = True
                    break
        
        self.assertTrue(has_clump, "Expected some clumping when diverse=false")

if __name__ == "__main__":
    unittest.main()
