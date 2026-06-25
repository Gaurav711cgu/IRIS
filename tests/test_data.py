import unittest
import numpy as np
from src.data_pipeline import SatelliteDatasetPreprocessor

class TestSatelliteDatasetPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = SatelliteDatasetPreprocessor()

    def test_scale_reflectance(self):
        """
        Verify that raw OLI digital numbers (typically 0-65535) scale
        properly within [0, 1] reflectance bounds.
        """
        raw_data = np.array([0, 10000, 20000, 50000], dtype=np.uint16)
        scaled = self.preprocessor.scale_reflectance(raw_data)
        
        self.assertTrue(np.all(scaled >= 0.0))
        self.assertTrue(np.all(scaled <= 1.0))
        # Check specific scaling calculation
        # DN * 0.0000275 - 0.2
        expected_first = max(0.0, 0.0 * 0.0000275 - 0.2)
        self.assertAlmostEqual(scaled[0], expected_first)

    def test_scale_thermal(self):
        """
        Verify that TIRS digital numbers scale correctly to Kelvin temperatures.
        """
        raw_data = np.array([30000, 40000], dtype=np.uint16)
        scaled = self.preprocessor.scale_thermal(raw_data)
        
        # Check temperature ranges (~250K to ~285K for these values)
        self.assertTrue(np.all(scaled > 200.0))
        self.assertTrue(np.all(scaled < 350.0))
        # Formula: DN * 0.00341802 + 149.0
        self.assertAlmostEqual(scaled[0], 30000 * 0.00341802 + 149.0, places=4)

    def test_calculate_ndvi(self):
        """
        Verify NDVI calculation math and division-by-zero safety.
        """
        b5 = np.array([0.5, 0.8, 0.0], dtype=np.float32)
        b4 = np.array([0.1, 0.2, 0.0], dtype=np.float32)
        ndvi = self.preprocessor.calculate_ndvi(b5, b4)
        
        # (0.5 - 0.1) / (0.5 + 0.1) = 0.4 / 0.6 = 0.6666
        self.assertAlmostEqual(ndvi[0], 0.6666667, places=5)
        self.assertTrue(np.all(ndvi >= -1.0))
        self.assertTrue(np.all(ndvi <= 1.0))
        # Ensure division by zero doesn't crash or return NaN/Inf
        self.assertFalse(np.isnan(ndvi[2]))
        self.assertFalse(np.isinf(ndvi[2]))

    def test_calculate_ndwi(self):
        """
        Verify NDWI calculation math and division-by-zero safety.
        """
        b3 = np.array([0.6, 0.1, 0.0], dtype=np.float32)
        b5 = np.array([0.2, 0.9, 0.0], dtype=np.float32)
        ndwi = self.preprocessor.calculate_ndwi(b3, b5)
        
        # (0.6 - 0.2) / (0.6 + 0.2) = 0.4 / 0.8 = 0.5
        self.assertAlmostEqual(ndwi[0], 0.5, places=5)
        self.assertTrue(np.all(ndwi >= -1.0))
        self.assertTrue(np.all(ndwi <= 1.0))
        self.assertFalse(np.isnan(ndwi[2]))
        self.assertFalse(np.isinf(ndwi[2]))

    def test_downsample_band(self):
        """
        Verify that 2x downsampling scales shapes by exactly 0.5.
        """
        mock_grid = np.random.rand(100, 100).astype(np.float32)
        downscaled = self.preprocessor.downsample_band(mock_grid, scale_factor=0.5)
        
        self.assertEqual(downscaled.shape, (50, 50))

if __name__ == '__main__':
    unittest.main()
