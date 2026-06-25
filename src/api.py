import os
import io
import hashlib
import numpy as np
import torch
import redis
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import StreamingResponse
from PIL import Image
from sqlalchemy.orm import Session

# Import database models
from src.database import init_db, get_db, ProcessedScene, YoloDetection
from src.models import GeneratorRRDB

# Try importing YOLO from ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

app = FastAPI(title="Project IRIS - Satellite Thermal Super-Resolution & Colorization API")

MODEL_PATH = "models/generator.pt"

# 1. Initialize models and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net_G = GeneratorRRDB(in_channels=6, out_channels=3)

if os.path.exists(MODEL_PATH):
    net_G.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"FastAPI: Loaded generator weights from {MODEL_PATH}")
else:
    print("FastAPI WARNING: Model weights not found. Running with random weights.")
net_G.to(device)
net_G.eval()

# 2. Setup Redis Cache with local in-memory fallback
REDIS_URL = os.getenv("REDIS_URL")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
redis_client = None
redis_available = False
in_memory_cache = {}

try:
    if REDIS_URL:
        redis_client = redis.Redis.from_url(REDIS_URL, socket_timeout=2)
        print("FastAPI: Attempting connection via REDIS_URL...")
    else:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, socket_timeout=2)
        print(f"FastAPI: Attempting connection via host/port at {REDIS_HOST}:{REDIS_PORT}...")
    redis_client.ping()
    redis_available = True
    print("FastAPI: Successfully connected to Redis Cache.")
except Exception as e:
    print(f"FastAPI WARNING: Redis connection failed ({str(e)}). Falling back to local in-memory dictionary cache.")

# 3. Database Initialization on Startup
@app.on_event("startup")
def startup_event():
    init_db()
    print("FastAPI: Initialized Database Tables successfully.")

@app.post("/colorize")
async def colorize(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Accepts a 6-channel numpy file (.npy) representing the stacked low-res input:
    [LR Thermal, Red, Green, Blue, NDVI, NDWI].
    Returns the 2x upscaled pseudo-RGB image as a PNG stream.
    Caches the output image and logs metadata/YOLO detections in SQL database.
    """
    filename = file.filename
    if not filename.endswith(".npy"):
        raise HTTPException(status_code=400, detail="Only .npy files containing the 6-channel stack are supported.")
        
    try:
        # 1. Read uploaded file bytes and calculate MD5 Checksum
        file_bytes = await file.read()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        cache_key = f"iris:cache:{file_hash}"
        
        # 2. Check Cache
        cached_png = None
        if redis_available:
            try:
                cached_png = redis_client.get(cache_key)
            except Exception:
                pass
        else:
            cached_png = in_memory_cache.get(cache_key)
            
        if cached_png is not None:
            print(f"FastAPI Cache Hit: Served colorized tile from cache (MD5: {file_hash})")
            return StreamingResponse(io.BytesIO(cached_png), media_type="image/png")
            
        # Cache Miss: Process the file
        print(f"FastAPI Cache Miss: Processing tile (MD5: {file_hash})...")
        
        # 3. Load numpy array and run generator inference
        buf = io.BytesIO(file_bytes)
        input_stack = np.load(buf) # Shape: [6, H, W]
        
        if input_stack.ndim != 3 or input_stack.shape[0] != 6:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid array shape: {input_stack.shape}. Expected [6, H, W]."
            )
            
        input_tensor = torch.tensor(input_stack, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output_tensor = net_G(input_tensor) # Shape: [1, 3, 2H, 2W]
            output_arr = output_tensor.squeeze(0).cpu().numpy() # Shape: [3, 2H, 2W]
            
        # 4. Convert generated pseudo-RGB array [3, 2H, 2W] to PNG bytes
        rgb_img_arr = np.transpose(output_arr, (1, 2, 0))
        rgb_img_uint8 = (rgb_img_arr * 255.0).clip(0, 255).astype(np.uint8)
        
        pil_img = Image.fromarray(rgb_img_uint8)
        img_io = io.BytesIO()
        pil_img.save(img_io, format="PNG")
        png_bytes = img_io.getvalue()
        
        # 5. Extract statistics
        ndvi_mean = float(input_stack[4].mean())
        ndwi_mean = float(input_stack[5].mean())
        
        # 6. Save Scene Metadata to Database
        tile_id_val = f"tile_{file_hash[:10]}"
        raw_path = f"data/processed/{tile_id_val}_input.npy"
        colorized_path = f"data/processed/{tile_id_val}_colorized.png"
        
        # Cache raw files locally
        os.makedirs("data/processed", exist_ok=True)
        np.save(raw_path, input_stack)
        pil_img.save(colorized_path)
        
        scene_db = ProcessedScene(
            tile_id=tile_id_val,
            platform="landsat-8",
            ndvi_mean=ndvi_mean,
            ndwi_mean=ndwi_mean,
            raw_file_path=raw_path,
            colorized_file_path=colorized_path,
            bbox_min_lon=77.0, bbox_min_lat=28.4, bbox_max_lon=77.3, bbox_max_lat=28.7 # default bounds
        )
        db.add(scene_db)
        db.flush() # Flush to populate scene_db.id
        
        # 7. Run YOLOv8 on output PNG and store detections in database
        if YOLO_AVAILABLE:
            yolo_model = YOLO("yolov8n.pt")
            yolo_res = yolo_model(rgb_img_uint8, verbose=False)[0]
            
            boxes = yolo_res.boxes.xyxy.cpu().numpy()
            classes = yolo_res.boxes.cls.cpu().numpy()
            confs = yolo_res.boxes.conf.cpu().numpy()
            
            # Draw detections and add to database session
            for i, box in enumerate(boxes):
                cls_id = int(classes[i])
                cls_name = yolo_model.names.get(cls_id, f"obj_{cls_id}")
                
                det_db = YoloDetection(
                    scene_id=scene_db.id,
                    label=cls_name,
                    confidence=float(confs[i]),
                    x1=float(box[0]), y1=float(box[1]), x2=float(box[2]), y2=float(box[3])
                )
                db.add(det_db)
                
        db.commit()
        print(f"FastAPI Database Logged: Registered scene metadata and YOLO detections (ID: {scene_db.id})")
        
        # 8. Save PNG to Cache
        if redis_available:
            try:
                redis_client.setex(cache_key, 86400, png_bytes) # Expire in 24 hours
            except Exception:
                pass
        else:
            if len(in_memory_cache) >= 100:
                # Evict first key
                in_memory_cache.pop(next(iter(in_memory_cache)))
            in_memory_cache[cache_key] = png_bytes
            
        return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference/Database error: {str(e)}")

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": os.path.exists(MODEL_PATH),
        "device": str(device),
        "redis_connected": redis_available,
        "database_connected": True
    }
