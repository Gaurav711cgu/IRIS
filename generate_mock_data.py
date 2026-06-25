import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from src.data_pipeline import SatelliteDatasetPreprocessor

def create_mock_raster(path, width, height, resolution, dtype, val_range, nodata=None, crs="EPSG:32643"):
    transform = from_origin(700000, 3100000, resolution, resolution)
    meta = {
        'driver': 'GTiff',
        'dtype': dtype,
        'nodata': nodata,
        'width': width,
        'height': height,
        'count': 1,
        'crs': crs,
        'transform': transform
    }
    
    # Generate data
    if dtype == 'uint8':
        # For WorldCover classes: 10 (Trees), 20 (Shrubland), 50 (Built-up), 80 (Water)
        classes = [10, 20, 50, 80]
        data = np.random.choice(classes, size=(height, width)).astype(np.uint8)
    else:
        data = np.random.randint(val_range[0], val_range[1], size=(height, width)).astype(np.uint16)
        
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, 'w', **meta) as dst:
        dst.write(data, 1)
    print(f"Created mock raster: {path} (shape: {width}x{height}, resolution: {resolution}m)")

def main():
    print("Generating mock dataset for Project IRIS...")
    
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    
    # Landsat 30m bands: B2, B3, B4, B5, B6
    # 512x512 pixels covers ~15.3km x 15.3km
    bands_30m = ["B2", "B3", "B4", "B5", "B6"]
    for band in bands_30m:
        create_mock_raster(
            path=os.path.join(raw_dir, f"{band}.tif"),
            width=512,
            height=512,
            resolution=30,
            dtype='uint16',
            val_range=(8000, 15000) # Typical Landsat Level-2 DN values
        )
        
    # Landsat 100m band: B10
    # covers same area: 15.3km / 100m = ~154 pixels
    create_mock_raster(
        path=os.path.join(raw_dir, "B10.tif"),
        width=154,
        height=154,
        resolution=100,
        dtype='uint16',
        val_range=(35000, 45000) # Typical thermal band scaling range
    )
    
    # ESA WorldCover 10m band
    # covers same area: 15.3km / 10m = ~1536 pixels
    create_mock_raster(
        path=os.path.join(raw_dir, "worldcover.tif"),
        width=1536,
        height=1536,
        resolution=10,
        dtype='uint8',
        val_range=(0, 255)
    )
    
    print("\nRunning preprocessor on mock data...")
    preprocessor = SatelliteDatasetPreprocessor()
    
    bands_paths = {
        "B2": os.path.join(raw_dir, "B2.tif"),
        "B3": os.path.join(raw_dir, "B3.tif"),
        "B4": os.path.join(raw_dir, "B4.tif"),
        "B5": os.path.join(raw_dir, "B5.tif"),
        "B6": os.path.join(raw_dir, "B6.tif"),
        "B10": os.path.join(raw_dir, "B10.tif")
    }
    cover_path = os.path.join(raw_dir, "worldcover.tif")
    output_dir = "data/processed"
    
    shapes = preprocessor.process_tile(
        bands_paths=bands_paths,
        cover_path=cover_path,
        output_dir=output_dir,
        tile_id="delhi_ncr"
    )
    
    print("\nMock data pipeline processing complete! Output shapes:")
    for k, v in shapes.items():
        print(f" - {k}: {v}")

if __name__ == "__main__":
    main()
