import os
import pystac_client
import planetary_computer
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import Window
import requests

def crop_and_save_asset(asset_url, bbox_gps, output_path):
    """
    Open asset URL (Cloud-Optimized GeoTIFF) remotely, crop to bbox_gps,
    and save the crop locally as a GeoTIFF.
    bbox_gps: [min_lon, min_lat, max_lon, max_lat]
    """
    print(f"Opening remote asset for: {output_path}...")
    with rasterio.open(asset_url) as src:
        # Reproject GPS bounds to the raster's CRS
        min_lon, min_lat, max_lon, max_lat = bbox_gps
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, min_lon, min_lat, max_lon, max_lat)
        
        # Get pixel coordinates for window
        row_start, col_start = src.index(left, top)
        row_stop, col_stop = src.index(right, bottom)
        
        # Ensure coordinates are within image boundaries
        row_start = max(0, min(row_start, src.height - 1))
        row_stop = max(0, min(row_stop, src.height))
        col_start = max(0, min(col_start, src.width - 1))
        col_stop = max(0, min(col_stop, src.width))
        
        # Sort values in case coordinates are flipped
        row_start, row_stop = min(row_start, row_stop), max(row_start, row_stop)
        col_start, col_stop = min(col_start, col_stop), max(col_start, col_stop)
        
        # Define window
        width = col_stop - col_start
        height = row_stop - row_start
        
        if width <= 0 or height <= 0:
            raise ValueError(f"Calculated window size is invalid: {width}x{height}")
            
        window = Window(col_start, row_start, width, height)
        print(f"Reading window: {window} (size: {width}x{height}) from {src.width}x{src.height} source...")
        
        # Read the cropped data
        data = src.read(1, window=window)
        
        # Update metadata for output crop
        new_meta = src.meta.copy()
        new_transform = rasterio.windows.transform(window, src.transform)
        new_meta.update({
            "height": height,
            "width": width,
            "transform": new_transform
        })
        
        # Save output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with rasterio.open(output_path, "w", **new_meta) as dst:
            dst.write(data, 1)
        print(f"Saved cropped band to {output_path}")

def main():
    # 1. Bounding box around Delhi-NCR
    # GPS: [min_lon, min_lat, max_lon, max_lat]
    bbox_gps = [77.0, 28.4, 77.3, 28.7]
    
    # 2. Setup Planetary Computer STAC Client
    print("Connecting to Planetary Computer STAC API...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    # 3. Search for Landsat-8/9 Level-2 scene
    print("Searching for Landsat 8/9 Level-2 scenes...")
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=bbox_gps,
        datetime="2023-10-01/2023-11-30",
        query={
            "platform": {"in": ["landsat-8", "landsat-9"]}
        },
        max_items=20
    )
    
    items = search.item_collection()
    if not items:
        raise ValueError("No matching Landsat items found.")
    
    # Pick item with lowest cloud cover
    best_item = min(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))
    print(f"Selected Item: {best_item.id} with cloud cover: {best_item.properties.get('eo:cloud_cover')}%")
    
    # Map band names to STAC asset keys
    band_mapping = {
        "B2": "blue",
        "B3": "green",
        "B4": "red",
        "B5": "nir08",
        "B6": "swir16",
        "B10": "lwir11"
    }
    
    # 4. Crop and save Landsat bands
    raw_dir = "data/raw"
    for band_name, asset_key in band_mapping.items():
        if asset_key in best_item.assets:
            asset_url = best_item.assets[asset_key].href
            output_path = os.path.join(raw_dir, f"{band_name}.tif")
            crop_and_save_asset(asset_url, bbox_gps, output_path)
        else:
            print(f"WARNING: Asset key {asset_key} not found in Landsat item!")

    # 5. Search and download ESA WorldCover
    print("Searching for ESA WorldCover 10m...")
    wc_search = catalog.search(
        collections=["esa-worldcover"],
        bbox=bbox_gps
    )
    wc_items = wc_search.item_collection()
    if wc_items:
        wc_item = wc_items[0]
        print(f"Selected ESA WorldCover Item: {wc_item.id}")
        asset_url = wc_item.assets["map"].href
        output_path = os.path.join(raw_dir, "worldcover.tif")
        crop_and_save_asset(asset_url, bbox_gps, output_path)
    else:
        print("WARNING: ESA WorldCover tile not found for bounding box!")

    print("All downloads and cropping completed successfully!")

if __name__ == "__main__":
    main()
