FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies for GDAL, rasterio, and compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    gdal-bin \
    gcc \
    g++ \
    make \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for GDAL
ENV CPL_ZIP_SETLOCALE=ON \
    GDAL_DATA=/usr/share/gdal \
    PROJ_LIB=/usr/share/proj

# Copy requirements file
COPY requirements.txt .

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default command
CMD ["bash"]
