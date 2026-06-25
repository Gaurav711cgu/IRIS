import os
import numpy as np
import rasterio
import rasterio.warp
from rasterio.enums import Resampling

class SatelliteDatasetPreprocessor:
    """
    Geospatial Preprocessing Pipeline for Project IRIS.
    Performs radiometric scaling, Band 11 exclusion, NDVI/NDWI indexing,
    resampling of ESA WorldCover classification labels, and 2x bilinear thermal downsampling.
    """
    def __init__(self, target_crs="EPSG:32643"):
        self.target_crs = target_crs

    def scale_reflectance(self, data):
        """
        Scale Landsat 8/9 OLI visible/auxiliary bands (B2, B3, B4, B5, B6)
        to [0, 1] reflectance range using USGS Collection 2 metadata coefficients.
        """
        # Multiplicative scaling factor: 0.0000275, Additive offset: -0.2
        scaled = data.astype(np.float32) * 0.0000275 - 0.2
        return np.clip(scaled, 0.0, 1.0)

    def scale_thermal(self, data):
        """
        Scale Landsat 8/9 TIRS Band 10 to Brightness Temperature in Kelvin.
        """
        # Multiplicative scaling factor: 0.00341802, Additive offset: 149.0
        # Kelvin range typically ranges from ~200K to ~340K.
        scaled = data.astype(np.float32) * 0.00341802 + 149.0
        return scaled

    def calculate_ndvi(self, b5, b4):
        """
        Calculate Normalized Difference Vegetation Index (NDVI) safely.
        NDVI = (NIR - Red) / (NIR + Red) -> (B5 - B4) / (B5 + B4)
        """
        denom = b5 + b4
        denom[denom == 0.0] = 1e-5 # Prevent division by zero
        ndvi = (b5 - b4) / denom
        return np.clip(ndvi, -1.0, 1.0)

    def calculate_ndwi(self, b3, b5):
        """
        Calculate Normalized Difference Water Index (NDWI) safely.
        NDWI = (Green - NIR) / (Green + NIR) -> (B3 - B5) / (B3 + B5)
        """
        denom = b3 + b5
        denom[denom == 0.0] = 1e-5 # Prevent division by zero
        ndwi = (b3 - b5) / denom
        return np.clip(ndwi, -1.0, 1.0)

    def resample_raster(self, src_path, match_meta, resampling_method=Resampling.nearest):
        """
        Resample cover classification raster to match the geometry and spatial reference
        of Landsat grids (100m or 30m). Nearest neighbor is used to preserve discrete label IDs.
        """
        with rasterio.open(src_path) as src:
            target_shape = (match_meta['height'], match_meta['width'])
            data = np.zeros(target_shape, dtype=src.dtypes[0])

            rasterio.warp.reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=match_meta['transform'],
                dst_crs=match_meta['crs'],
                resampling=resampling_method
            )
            return data

    def downsample_band(self, data, scale_factor=0.5):
        """
        Bilinear downsamples high-resolution Band 10 grid (100m) to low-resolution grid (200m).
        Used to prepare inputs for evaluation protocol validation (200m -> 100m upscaling).
        """
        h, w = data.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        # Reshape for interpolation
        from scipy.ndimage import zoom
        downscaled = zoom(data, scale_factor, order=1) # order=1 is bilinear
        return downscaled

    def process_tile(self, bands_paths, cover_path, output_dir, tile_id):
        """
        Coordinates the ingestion, cleaning, normalization, indexing, and downsampling of a single tile.
        
        Parameters:
        - bands_paths: Dict containing keys: 'B2', 'B3', 'B4', 'B5', 'B6', 'B10'
                       (Band 11 is explicitly excluded by not including it in the pipeline)
        - cover_path: Path to the matching ESA WorldCover raster
        - output_dir: Directory where processed NumPy arrays will be cached
        - tile_id: Unique string identifier for the tile
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Load and scale bands
        scaled_bands = {}
        target_meta = None
        
        for band_name, path in bands_paths.items():
            with rasterio.open(path) as src:
                data = src.read(1)
                if band_name == 'B10':
                    scaled_bands[band_name] = self.scale_thermal(data)
                    target_meta = src.meta.copy() # Match dimensions based on B10 grid
                else:
                    scaled_bands[band_name] = self.scale_reflectance(data)

        # 2. Compute spectral indices
        ndvi = self.calculate_ndvi(scaled_bands['B5'], scaled_bands['B4'])
        ndwi = self.calculate_ndwi(scaled_bands['B3'], scaled_bands['B5'])

        # 3. Resample ESA WorldCover semantic classification labels to match target B10 metadata
        if cover_path and os.path.exists(cover_path):
            cover_data = self.resample_raster(cover_path, target_meta, Resampling.nearest)
        else:
            # Fallback placeholder if no cover classification is available
            cover_data = np.zeros((target_meta['height'], target_meta['width']), dtype=np.uint8)

        # 4. Generate 2x downsampled thermal input (200m)
        real_thermal_100m = scaled_bands['B10']
        low_res_thermal_200m = self.downsample_band(real_thermal_100m, scale_factor=0.5)

        # 5. Pack data into NumPy arrays
        # Generator Input Stack: LR Thermal, Red, Green, Blue, NDVI, NDWI
        # We resample all auxiliary/visible bands (30m) to match the LR Thermal grid size for training
        h_lr, w_lr = low_res_thermal_200m.shape
        
        from scipy.ndimage import zoom
        def resize_to_lr(arr):
            return zoom(arr, (h_lr / arr.shape[0], w_lr / arr.shape[1]), order=1)

        b4_lr = resize_to_lr(scaled_bands['B4'])
        b3_lr = resize_to_lr(scaled_bands['B3'])
        b2_lr = resize_to_lr(scaled_bands['B2'])
        ndvi_lr = resize_to_lr(ndvi)
        ndwi_lr = resize_to_lr(ndwi)

        input_stack = np.stack([
            low_res_thermal_200m,
            b4_lr,
            b3_lr,
            b2_lr,
            ndvi_lr,
            ndwi_lr
        ], axis=0) # Shape: [6, H_lr, W_lr]

        # Targets
        # Ground Truth Thermal (100m)
        target_thermal = real_thermal_100m # Shape: [H, W] (which is 2 * H_lr)
        # Ground Truth RGB color mapping
        # Resized to match target_thermal (100m grid) to avoid shape mismatch during loss calculation
        h_t, w_t = target_thermal.shape
        b4_t = zoom(scaled_bands['B4'], (h_t / scaled_bands['B4'].shape[0], w_t / scaled_bands['B4'].shape[1]), order=1)
        b3_t = zoom(scaled_bands['B3'], (h_t / scaled_bands['B3'].shape[0], w_t / scaled_bands['B3'].shape[1]), order=1)
        b2_t = zoom(scaled_bands['B2'], (h_t / scaled_bands['B2'].shape[0], w_t / scaled_bands['B2'].shape[1]), order=1)
        target_rgb = np.stack([b4_t, b3_t, b2_t], axis=0)

        # Save arrays
        np.save(os.path.join(output_dir, f"{tile_id}_input.npy"), input_stack)
        np.save(os.path.join(output_dir, f"{tile_id}_target_thermal.npy"), target_thermal)
        np.save(os.path.join(output_dir, f"{tile_id}_target_rgb.npy"), target_rgb)
        np.save(os.path.join(output_dir, f"{tile_id}_cover_mask.npy"), cover_data)

        print(f"Successfully processed and saved arrays for tile: {tile_id}")
        return {
            "input_stack_shape": input_stack.shape,
            "target_thermal_shape": target_thermal.shape,
            "target_rgb_shape": target_rgb.shape,
            "cover_mask_shape": cover_data.shape
        }
