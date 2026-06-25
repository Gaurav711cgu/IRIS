import os
import io
import time
import numpy as np
import requests
import torch
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from src.models import GeneratorRRDB

# Try importing YOLO from ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Project IRIS - Satellite Super-Resolution & Colorization",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for premium dark-mode aesthetics
st.markdown("""
<style>
    /* Main body background and text */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #151a24 100%);
        color: #e6ebf5;
        font-family: 'Outfit', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header card */
    .header-container {
        padding: 2rem 0;
        background: rgba(30, 41, 59, 0.4);
        border-radius: 12px;
        margin-bottom: 2rem;
        border-left: 5px solid #3b82f6;
        padding-left: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .header-title {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
    }
    
    /* Custom subheaders */
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #60a5fa;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 1. Sidebar Configurations
st.sidebar.markdown("<h2 style='color:#3b82f6;'>Config Panel</h2>", unsafe_allow_html=True)

api_url = st.sidebar.text_input("FastAPI Colorization URL", value="http://localhost:8000/colorize")
st.sidebar.markdown("---")

load_test = st.sidebar.checkbox("Load Local Test File (Delhi-NCR)", value=False)
st.sidebar.markdown("---")

yolo_toggle = st.sidebar.checkbox("Enable YOLOv8 Object Detection Overlay", value=True)
yolo_conf = st.sidebar.slider("YOLOv8 Confidence Threshold", min_value=0.1, max_value=1.0, value=0.25, step=0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Pipeline Info
*   **Input**: 6-channel Stacked Tensor [LR Thermal, Red, Green, Blue, NDVI, NDWI]
*   **Resolution**: 2x Super-Resolution (200m -> 100m)
*   **Models**: ESRGAN Generator (RRDB) & PatchGAN Discriminator
""")

# 2. Main Title Banner
st.markdown("""
<div class="header-container">
    <div class="header-title">Project IRIS</div>
    <div class="header-subtitle">Satellite Thermal Imagery 2× Super-Resolution and Pseudo-RGB Colorization</div>
</div>
""", unsafe_allow_html=True)

# Helper function to run inference locally if API is offline
def run_local_inference(input_stack, model_path="models/generator.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_G = GeneratorRRDB(in_channels=6, out_channels=3)
    if os.path.exists(model_path):
        net_G.load_state_dict(torch.load(model_path, map_location=device))
    net_G.to(device)
    net_G.eval()
    
    input_tensor = torch.tensor(input_stack, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output_tensor = net_G(input_tensor)
        output_arr = output_tensor.squeeze(0).cpu().numpy()
    
    rgb_img_arr = np.transpose(output_arr, (1, 2, 0))
    rgb_img_uint8 = (rgb_img_arr * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgb_img_uint8)

# Helper function to draw boxes on image
def draw_detections(image, boxes, classes, confs, conf_threshold):
    draw = ImageDraw.Draw(image)
    # Simple color map
    colors = ["#f87171", "#fbbf24", "#34d399", "#60a5fa", "#a78bfa", "#f472b6"]
    
    detected_counts = {}
    
    for i, box in enumerate(boxes):
        conf = confs[i]
        if conf < conf_threshold:
            continue
            
        cls_id = int(classes[i])
        cls_name = f"obj_{cls_id}"
        if YOLO_AVAILABLE:
            # We can use standard COCO class names if available
            yolo_dummy = YOLO("yolov8n.pt")
            cls_name = yolo_dummy.names.get(cls_id, cls_name)
            
        detected_counts[cls_name] = detected_counts.get(cls_name, 0) + 1
        
        # Select color based on class ID
        color = colors[cls_id % len(colors)]
        
        # Draw box outline
        draw.rectangle(list(box), outline=color, width=3)
        # Label text
        label = f"{cls_name} {conf:.2f}"
        draw.text((box[0] + 5, box[1] + 5), label, fill=color)
        
    return image, detected_counts

# 3. File Upload Interface
st.markdown("<div class='section-title'>Upload Satellite Stack</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a 6-channel preprocessed .npy file", type=["npy"])

input_stack = None

if load_test:
    test_file_path = "data/processed/delhi_ncr_input.npy"
    if os.path.exists(test_file_path):
        with open(test_file_path, "rb") as f:
            file_bytes = f.read()
        buf = io.BytesIO(file_bytes)
        input_stack = np.load(buf)
        st.success(f"Successfully loaded default test tile from disk: {test_file_path}")
    else:
        st.error(f"Default test file {test_file_path} not found. Please run generate_mock_data.py first.")
elif uploaded_file is not None:
    file_bytes = uploaded_file.read()
    buf = io.BytesIO(file_bytes)
    input_stack = np.load(buf)
    st.success(f"Successfully loaded satellite tile of shape: {input_stack.shape}")

if input_stack is not None:
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='section-title'>Raw Inputs</div>", unsafe_allow_html=True)
            # Visualize raw thermal input (first channel)
            raw_thermal = input_stack[0]
            # Normalize thermal to relative range for visual scaling
            t_min, t_max = raw_thermal.min(), raw_thermal.max()
            thermal_vis = (raw_thermal - t_min) / (t_max - t_min + 1e-5)
            st.image(thermal_vis, caption=f"Raw Low-Res Thermal (200m) - Temp Range: {t_min:.1f}K to {t_max:.1f}K", use_column_width=True)
            
            # Show NDVI index (fifth channel)
            ndvi_vis = (input_stack[4] + 1.0) / 2.0 # map [-1,1] to [0,1]
            st.image(ndvi_vis, caption="NDVI Vegetation Index (30m scaled)", use_column_width=True)

        with col2:
            st.markdown("<div class='section-title'>Colorized Inference Output</div>", unsafe_allow_html=True)
            
            # Action Button
            if st.button("Run Colorization Pipeline", type="primary"):
                with st.spinner("Processing tile through Multi-Objective GAN..."):
                    start_time = time.time()
                    colorized_img = None
                    used_api = False
                    
                    # 1. Attempt FastAPI colorization
                    try:
                        # Reset file buffer
                        buf.seek(0)
                        files = {"file": ("tile.npy", buf, "application/octet-stream")}
                        response = requests.post(api_url, files=files, timeout=5)
                        if response.status_code == 200:
                            colorized_img = Image.open(io.BytesIO(response.content))
                            used_api = True
                    except Exception as e:
                        pass
                    
                    # 2. Standalone fallback (run locally)
                    if colorized_img is None:
                        colorized_img = run_local_inference(input_stack)
                        
                    elapsed_time = time.time() - start_time
                    
                    # 3. Post-Process with YOLO Object Detection
                    if yolo_toggle and YOLO_AVAILABLE:
                        yolo_model = YOLO("yolov8n.pt")
                        # Convert PIL Image to numpy for YOLO
                        yolo_in = np.array(colorized_img)
                        res = yolo_model(yolo_in, verbose=False)[0]
                        
                        boxes = res.boxes.xyxy.cpu().numpy()
                        classes = res.boxes.cls.cpu().numpy()
                        confs = res.boxes.conf.cpu().numpy()
                        
                        # Draw boxes on PIL Image
                        colorized_img, counts = draw_detections(colorized_img, boxes, classes, confs, yolo_conf)
                        
                        st.image(colorized_img, caption="Upscaled Colorized Thermal (100m) with YOLO Detection Bboxes", use_column_width=True)
                        
                        # Print detected counts in metrics
                        st.write("### Detections Statistics")
                        if counts:
                            st.write(counts)
                        else:
                            st.write("No objects detected above threshold.")
                    else:
                        st.image(colorized_img, caption="Upscaled Colorized Thermal (100m)", use_column_width=True)
                        
                    st.metric(
                        label="Pipeline Inference Time",
                        value=f"{elapsed_time:.3f} seconds",
                        delta="API Mode" if used_api else "Local Fallback Mode"
                    )
else:
    st.info("Please upload a preprocessed satellite tile array (.npy) using the box above to run the pipeline.")
