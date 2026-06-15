import os
import unittest
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "wind_turbine_suitability")
MONGO_GRID_COLLECTION = os.getenv("MONGO_GRID_COLLECTION", "grid_cells")
MONGO_STATIONS_COLLECTION = os.getenv("MONGO_STATIONS_COLLECTION", "knmi_stations")
MONGO_MODEL_RUNS_COLLECTION = os.getenv("MONGO_MODEL_RUNS_COLLECTION", "model_runs")

class TestMongoDBIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        # Verify connection
        try:
            cls.client.server_info()
            cls.db_online = True
            cls.db = cls.client[MONGO_DB]
        except Exception as e:
            cls.db_online = False
            print(f"\n⚠️ WARNING: Could not connect to MongoDB at {MONGO_URI}. Some tests will be skipped. Error: {e}")

    def setUp(self):
        if not self.db_online:
            self.skipTest("MongoDB connection offline")

    def test_connection_and_db_exists(self):
        """Verify that the connection is active and database exists."""
        self.assertTrue(self.db_online)
        db_names = self.client.list_database_names()
        # MongoDB creates a database only when a document is inserted, which we expect to be done
        self.assertIn(MONGO_DB, db_names, f"Database '{MONGO_DB}' not found in active database names!")

    def test_grid_cells_count(self):
        """Verify that grid_cells count is around 17,147."""
        grid_col = self.db[MONGO_GRID_COLLECTION]
        count = grid_col.count_documents({})
        print(f"\n📊 [Test] Active Grid Cells Count: {count}")
        # Allow small deviation in dataset size just in case, but around 17,147
        self.assertTrue(17000 <= count <= 17300, f"Grid cells count is {count}, expected around 17,147")

    def test_knmi_stations_count(self):
        """Verify that knmi_stations count is greater than 0."""
        stations_col = self.db[MONGO_STATIONS_COLLECTION]
        count = stations_col.count_documents({})
        print(f"📊 [Test] Active KNMI Stations Count: {count}")
        self.assertGreater(count, 0, "No KNMI stations found in database!")
        self.assertTrue(45 <= count <= 55, f"Standard Netherlands stations count should be around 50, got {count}")

    def test_indexes_exist(self):
        """Verify that all required indexes exist on the collections."""
        grid_col = self.db[MONGO_GRID_COLLECTION]
        stations_col = self.db[MONGO_STATIONS_COLLECTION]
        runs_col = self.db[MONGO_MODEL_RUNS_COLLECTION]
        
        # Get index names/keys
        grid_indexes = grid_col.index_information()
        stations_indexes = stations_col.index_information()
        runs_indexes = runs_col.index_information()
        
        # Helper to check key existence in indexes
        def has_index_on_keys(indexes_info, expected_keys):
            for idx_name, idx_details in indexes_info.items():
                idx_keys = idx_details["key"]
                if idx_keys == expected_keys:
                    return True
            return False
            
        # Verify grid_cells indexes
        self.assertTrue(has_index_on_keys(grid_indexes, [("location", "2dsphere")]), "Missing 'location' 2dsphere index on grid_cells!")
        self.assertTrue(has_index_on_keys(grid_indexes, [("cell_lat", 1), ("cell_lon", 1)]), "Missing '(cell_lat, cell_lon)' index on grid_cells!")
        self.assertTrue(has_index_on_keys(grid_indexes, [("scores.ml_suitability_score", -1)]), "Missing 'scores.ml_suitability_score' descending index on grid_cells!")
        self.assertTrue(has_index_on_keys(grid_indexes, [("display.ml_suitable", 1), ("scores.ml_suitability_score", -1)]), "Missing compound index on grid_cells!")
        self.assertTrue(has_index_on_keys(grid_indexes, [("features.is_natura2000", 1)]), "Missing 'features.is_natura2000' index on grid_cells!")
        self.assertTrue(has_index_on_keys(grid_indexes, [("kmeans.label", 1)]), "Missing 'kmeans.label' index on grid_cells!")
        
        # Verify stations indexes
        self.assertTrue(has_index_on_keys(stations_indexes, [("station_id", 1)]), "Missing 'station_id' index on knmi_stations!")
        self.assertTrue(has_index_on_keys(stations_indexes, [("location", "2dsphere")]), "Missing 'location' 2dsphere index on knmi_stations!")
        
        # Verify runs index
        self.assertTrue(has_index_on_keys(runs_indexes, [("run_id", 1)]), "Missing 'run_id' index on model_runs!")

    def test_query_high_suitability(self):
        """Verify that we can query highly suitable cells (ml_suitability_score >= 70)."""
        grid_col = self.db[MONGO_GRID_COLLECTION]
        high_suitability_cells = list(grid_col.find({"scores.ml_suitability_score": {"$gte": 70.0}}).limit(5))
        
        self.assertGreater(len(high_suitability_cells), 0, "No highly suitable cells found in database!")
        for cell in high_suitability_cells:
            self.assertGreaterEqual(cell["scores"]["ml_suitability_score"], 70.0)
            self.assertEqual(cell["display"]["very_suitable"], True)

    def test_bounding_box_query(self):
        """Verify that a bounding box query returns valid grid-cell coordinates."""
        grid_col = self.db[MONGO_GRID_COLLECTION]
        # Bounding box near Central Netherlands (Utrecht/Amsterdam area)
        min_lat, max_lat = 52.0, 52.5
        min_lon, max_lon = 4.7, 5.2
        
        query = {
            "cell_lat": {"$gte": min_lat, "$lte": max_lat},
            "cell_lon": {"$gte": min_lon, "$lte": max_lon}
        }
        
        bbox_results = list(grid_col.find(query).limit(10))
        self.assertGreater(len(bbox_results), 0, "No bounding box results found!")
        for cell in bbox_results:
            self.assertTrue(min_lat <= cell["cell_lat"] <= max_lat)
            self.assertTrue(min_lon <= cell["cell_lon"] <= max_lon)

    def test_latest_model_run_exists(self):
        """Verify that the latest model run report is saved and retrieved."""
        runs_col = self.db[MONGO_MODEL_RUNS_COLLECTION]
        latest_run = runs_col.find_one(sort=[("timestamp", -1)])
        
        self.assertIsNotNone(latest_run, "No model runs metadata found in collection!")
        self.assertIn("run_id", latest_run)
        self.assertIn("timestamp", latest_run)
        self.assertIn("metrics", latest_run)
        self.assertIn("kmeans", latest_run["metrics"])

if __name__ == "__main__":
    unittest.main()
