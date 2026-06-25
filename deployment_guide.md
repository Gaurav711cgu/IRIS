# Project IRIS - Production Cloud Deployment Guide

This guide provides step-by-step instructions to deploy the static frontend to **Vercel**, host the FastAPI backend on a free **Google Colab T4 GPU**, and connect remote cloud databases (**Supabase PostgreSQL** and **Upstash Redis**).

---

## 1. Cloud Database Integration

### A. Supabase PostgreSQL (Relational Metadata Store)
We use a relational database to save scene metadata, vegetation/water indices, and YOLO object detections.

1. Sign up for a free account at [Supabase](https://supabase.com).
2. Click **New Project** and configure your database name, region, and a secure **Database Password** (save this password).
3. Wait for the database instance to provision (typically 1–2 minutes).
4. Go to **Project Settings** (gear icon) &rarr; **Database**.
5. Scroll down to the **Connection string** section, select the **URI** tab, and copy the connection string.
   * *Example string:* `postgresql://postgres.[your-project-id]:[your-password]@aws-0-[region].pooler.supabase.com:6543/postgres`
6. Replace `[your-password]` with the actual database password you created.
7. To configure the backend to use this database, set the connection string as an environment variable:
   ```bash
   export DATABASE_URL="postgresql://postgres.[your-project-id]:[your-password]@aws-0-[region].pooler.supabase.com:6543/postgres"
   ```

### B. Upstash Redis (In-Memory Image Caching)
We use a Redis database to cache super-resolved output files based on their MD5 payload checksums, rendering repeat tiles in sub-5 milliseconds.

1. Sign up for a free account at [Upstash](https://upstash.com).
2. Click **Create Database**, select **Redis**, choose a region close to your Colab backend/user, and click **Create**.
3. Under database details, locate the **Connect to your database** section and copy the **Redis URL** (under the `.env` tab or node settings).
   * *Example string:* `rediss://default:[your-password]@[your-endpoint].upstash.io:6379`
4. To configure the backend to use this cache, set the URL as an environment variable:
   ```bash
   export REDIS_URL="rediss://default:[your-password]@[your-endpoint].upstash.io:6379"
   ```

### C. Verify Connections
Run our connection diagnostic script on your host machine to verify that your credentials are correct and accessible:
```bash
# Set your environment variables
export REDIS_URL="rediss://default:..."
export DATABASE_URL="postgresql://postgres..."

# Run the test
PYTHONPATH=. python3 scratch/verify_cloud_db.py
```

---

## 2. GPU-Accelerated Backend Hosting (Google Colab)

Google Colab offers a free Nvidia T4 GPU, which provides the computing power required to execute our ESRGAN and YOLOv8 models.

### Steps to Run:
1. Open a new notebook on [Google Colab](https://colab.research.google.com).
2. Set the runtime to use a GPU: Go to **Runtime** &rarr; **Change runtime type** &rarr; select **T4 GPU** &rarr; click **Save**.
3. Copy and run the following cells inside the Colab notebook:

#### Cell 1: Clone Repository & Install Requirements
```python
# 1. Clone the project repository
!git clone https://github.com/Gaurav711cgu/IRIS.git
%cd IRIS

# 2. Install backend dependencies
!pip install -r requirements.txt
!pip install uvicorn
```

#### Cell 2: Configure Environment & Run API with Public Tunnel
```python
import os
import subprocess
import time

# --- A. Setup Cloud DB Variables (Replace with your actual keys) ---
os.environ["PYTHONPATH"] = os.getcwd()
os.environ["DATABASE_URL"] = "postgresql://postgres:ISROIRIS%40250626@db.prcufhzhaxsjsdjzbrks.supabase.co:5432/postgres"
os.environ["REDIS_URL"] = "rediss://default:gQAAAAAAAac4AAIgcDJjZWQ4NWQ3M2Q5ZTI0ZDEyYWM3ZGRlMjc1ZjAyZjRkOQ@renewing-antelope-108344.upstash.io:6379"

# --- B. Launch FastAPI Server in Background ---
print("Starting FastAPI Backend...")
log_file = open("fastapi_server.log", "w")
server_proc = subprocess.Popen(
    ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"],
    stdout=log_file, stderr=log_file
)
time.sleep(5)  # Wait for server to load weights

# --- C. Setup Free Localtunnel Forwarding ---
print("Configuring Localtunnel Port Forwarder...")
# Install localtunnel NPM package
!npm install -g localtunnel

# Start the tunnel on port 8000
tunnel_proc = subprocess.Popen(
    ["lt", "--port", "8000"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True
)

time.sleep(3)
# Read tunnel output to print your secure public URL
import urllib.request
# Get IP of Colab node to bypass localtunnel warning page
colab_ip = urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip()
print(f"\n=======================================================")
print(f"1. Your Colab IP for Bypass Page is: {colab_ip}")
print(f"2. Check localtunnel.log or stdout for your public URL.")
print(f"=======================================================\n")
```

#### Cell 3: View Public URL
```python
# Run this to print the active public localtunnel url
!curl https://localtunnel.me
!npx localtunnel --port 8000
```
*Copy the public URL (e.g. `https://dry-rivers-run.localtunnel.me`) generated by localtunnel. You will input this URL into your web dashboard settings.*

---

## 3. Web Dashboard Frontend Deployment (Vercel)

The web dashboard is a static glassmorphic frontend interface that requires zero server-side configurations, making it compatible with Vercel's free static hosting tier.

1. Push your local `project-iris` directory to a new **GitHub** repository.
2. Sign in to your [Vercel Dashboard](https://vercel.com).
3. Click **Add New** &rarr; **Project**.
4. Import your newly created GitHub repository.
5. In the configuration settings:
   * **Root Directory**: Click "Edit" and select the `frontend` subdirectory (so Vercel only deploys the static files).
   * **Framework Preset**: Choose **Other** or leave it as default.
   * **Build & Development Settings**: Keep default (no build script required).
6. Click **Deploy**.
7. Once deployed, click on your public Vercel URL to open the dashboard.

---

## 4. Link Dashboard to GPU Backend

1. Open your deployed web dashboard URL on Vercel.
2. Click the **API Settings** (Gear icon) in the top-right corner of the navbar.
3. In the slide-out connectivity pane:
   * Paste your Colab public URL (e.g. `https://[your-subdomain].localtunnel.me/colorize`) in the **FastAPI Colorize Endpoint** field.
   * *Note: Ensure the `/colorize` suffix is attached to your URL.*
4. Click **Test Connection**.
5. Once a connection is established successfully, the status indicator will turn green and read **GPU Online (CUDA)**.
6. Toggle the source mode to **Live GPU API** to upload custom `.npy` tiles, run real-time super-resolution, calculate live NDVI/NDWI metrics, detect targets with YOLOv8, and view database entries on the fly!
