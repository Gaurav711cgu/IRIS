import os
import shutil
import numpy as np
from PIL import Image
from scipy.ndimage import zoom

def save_image_from_array(arr, path):
    # arr is [3, H, W] in [0, 1] range
    arr_t = np.transpose(arr, (1, 2, 0)) # [H, W, 3]
    arr_uint8 = (arr_t * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr_uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    print(f"Saved image to: {path}")

def main():
    assets_dir = "frontend/assets"
    os.makedirs(assets_dir, exist_ok=True)
    
    # 1. Load preprocessed input stack
    input_path = "data/processed/delhi_ncr_input.npy"
    target_rgb_path = "data/processed/delhi_ncr_target_rgb.npy"
    test_output_png = "data/processed/test_output.png"
    
    if not (os.path.exists(input_path) and os.path.exists(target_rgb_path)):
        print("Preprocessed files not found. Please run generate_mock_data.py first.")
        return
        
    inputs = np.load(input_path)
    targets_rgb = np.load(target_rgb_path)
    
    # 2. Save Target RGB
    save_image_from_array(targets_rgb, os.path.join(assets_dir, "target_rgb.png"))
    
    # 3. Create and Save Raw Thermal
    low_res_thermal = inputs[0] # first channel -> [H_lr, W_lr]
    h_t, w_t = targets_rgb.shape[1], targets_rgb.shape[2]
    # Normalize
    t_min, t_max = low_res_thermal.min(), low_res_thermal.max()
    norm_thermal = (low_res_thermal - t_min) / (t_max - t_min + 1e-5)
    # Upscale 2x using bilinear interpolation
    thermal_upscaled = zoom(norm_thermal, (h_t / low_res_thermal.shape[0], w_t / low_res_thermal.shape[1]), order=1)
    thermal_3ch = np.stack([thermal_upscaled, thermal_upscaled, thermal_upscaled], axis=0)
    save_image_from_array(thermal_3ch, os.path.join(assets_dir, "raw_thermal.png"))
    
    # 4. Copy generated output
    if os.path.exists(test_output_png):
        shutil.copy(test_output_png, os.path.join(assets_dir, "colorized_output.png"))
        print(f"Copied test_output.png to: {os.path.join(assets_dir, 'colorized_output.png')}")
    else:
        # Fallback to random image if test_output doesn't exist
        print("Warning: test_output.png not found, creating random colorized fallback.")
        fake_color = np.random.rand(3, h_t, w_t)
        save_image_from_array(fake_color, os.path.join(assets_dir, "colorized_output.png"))

if __name__ == "__main__":
    main()
