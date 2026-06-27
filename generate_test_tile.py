import os
import numpy as np

def generate_sample_tiles():
    # Ensure raw directory exists
    os.makedirs("data/raw", exist_ok=True)
    
    # 1. Generate Delhi Sample (urban areas, mixed vegetation/water)
    # Shape: [6, 256, 256] -> [LR Thermal, Red, Green, Blue, NDVI, NDWI]
    delhi = np.random.rand(6, 256, 256).astype(np.float32)
    # Calibrate channels to typical ranges
    delhi[0] = 0.3 + 0.4 * delhi[0]  # Normalised Thermal temperature
    delhi[1] = 0.15 + 0.1 * delhi[1] # Red reflectance
    delhi[2] = 0.18 + 0.08 * delhi[2] # Green reflectance
    delhi[3] = 0.12 + 0.12 * delhi[3] # Blue reflectance
    delhi[4] = 0.2 + 0.6 * delhi[4]   # NDVI (positive vegetation)
    delhi[5] = -0.4 + 0.3 * delhi[5]  # NDWI (negative urban/dry land)
    
    np.save("data/raw/delhi_sample.npy", delhi)
    print("Generated: data/raw/delhi_sample.npy")

    # 2. Generate Crop Sample (high NDVI, negative NDWI)
    crop = np.random.rand(6, 256, 256).astype(np.float32)
    crop[0] = 0.2 + 0.3 * crop[0]
    crop[1] = 0.08 + 0.05 * crop[1]
    crop[2] = 0.25 + 0.15 * crop[2]
    crop[3] = 0.05 + 0.05 * crop[3]
    crop[4] = 0.6 + 0.3 * crop[4]    # High vegetation NDVI (dense canopy)
    crop[5] = -0.6 + 0.2 * crop[5]   # Strongly negative NDWI
    
    np.save("data/raw/crop_sample.npy", crop)
    print("Generated: data/raw/crop_sample.npy")

    # 3. Generate Water Body Sample (negative NDVI, positive NDWI)
    water = np.random.rand(6, 256, 256).astype(np.float32)
    water[0] = 0.1 + 0.2 * water[0]
    water[1] = 0.05 + 0.05 * water[1]
    water[2] = 0.1 + 0.1 * water[2]
    water[3] = 0.25 + 0.2 * water[3]
    water[4] = -0.3 + 0.2 * water[4]  # Negative NDVI for water
    water[5] = 0.3 + 0.5 * water[5]   # Positive NDWI (open water body)
    
    np.save("data/raw/water_sample.npy", water)
    print("Generated: data/raw/water_sample.npy")

if __name__ == "__main__":
    generate_sample_tiles()
