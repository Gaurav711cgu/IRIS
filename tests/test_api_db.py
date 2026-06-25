import unittest
import os
import io
import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from src.api import app
from src.database import SessionLocal, ProcessedScene, YoloDetection, engine, init_db

class TestApiDbIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create database tables for testing
        init_db()
        cls.client = TestClient(app)
        cls.db = SessionLocal()

    @classmethod
    def tearDownClass(cls):
        cls.db.close()
        # Clean up database file after testing
        if os.path.exists("./iris.db"):
            try:
                os.remove("./iris.db")
            except Exception:
                pass

    def test_health_endpoint(self):
        """
        Verify the FastAPI health check indicates standard services are healthy.
        """
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["database_connected"])

    def test_colorize_and_database_logging(self):
        """
        Verify that uploading a mock 6-channel stack colorizes successfully,
        saves metadata inside SQLite, and subsequent duplicate requests hit the cache.
        """
        # 1. Create mock 6-channel satellite stack [6, 77, 77]
        mock_stack = np.random.rand(6, 77, 77).astype(np.float32)
        # Populate NDVI/NDWI channels
        mock_stack[4] = 0.5  # NDVI
        mock_stack[5] = -0.2 # NDWI
        
        # Save to buffer
        buf = io.BytesIO()
        np.save(buf, mock_stack)
        buf.seek(0)
        
        # 2. Post file to colorize endpoint
        files = {'file': ('tile.npy', buf, 'application/octet-stream')}
        response = self.client.post("/colorize", files=files)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        
        # 3. Query the database to verify logging occurred
        scene = self.db.query(ProcessedScene).order_by(ProcessedScene.id.desc()).first()
        self.assertIsNotNone(scene)
        self.assertTrue(scene.tile_id.startswith("tile_"))
        self.assertAlmostEqual(scene.ndvi_mean, 0.5, places=5)
        self.assertAlmostEqual(scene.ndwi_mean, -0.2, places=5)
        
        # 4. Repeat the request with the same file payload to trigger cache hit
        buf.seek(0)
        files_repeat = {'file': ('tile.npy', buf, 'application/octet-stream')}
        response_repeat = self.client.post("/colorize", files=files_repeat)
        self.assertEqual(response_repeat.status_code, 200)
        
        # Assert database has not added a duplicate row (shows cache worked)
        scene_count = self.db.query(ProcessedScene).filter(ProcessedScene.tile_id == scene.tile_id).count()
        self.assertEqual(scene_count, 1)

if __name__ == '__main__':
    unittest.main()
