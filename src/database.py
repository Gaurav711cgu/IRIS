import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Switch to PostgreSQL if DATABASE_URL environment variable is provided, otherwise fallback to local SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./iris.db")

# For SQLite, we disable check_same_thread warning for FastAPI compatibility
connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class ProcessedScene(Base):
    """
    Metadata representation of processed satellite scenes.
    Saves spectral ranges, spatial bounding box, and file paths.
    """
    __tablename__ = "processed_scenes"

    id = Column(Integer, primary_key=True, index=True)
    tile_id = Column(String, unique=True, index=True, nullable=False)
    platform = Column(String, default="landsat-8")
    acquisition_date = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, default=datetime.utcnow)
    cloud_cover = Column(Float, default=0.0)
    
    # Statistical analysis values
    ndvi_mean = Column(Float, default=0.0)
    ndwi_mean = Column(Float, default=0.0)
    
    # File storage paths
    raw_file_path = Column(String, nullable=False)
    colorized_file_path = Column(String, nullable=False)
    
    # Geographic Bounding Box (cross-compatible Float coordinates for SQLite support)
    bbox_min_lon = Column(Float, nullable=True)
    bbox_min_lat = Column(Float, nullable=True)
    bbox_max_lon = Column(Float, nullable=True)
    bbox_max_lat = Column(Float, nullable=True)
    
    # Relationships
    detections = relationship("YoloDetection", back_populates="scene", cascade="all, delete-orphan")


class YoloDetection(Base):
    """
    Represents objects detected by YOLOv8 on the colorized output.
    """
    __tablename__ = "yolo_detections"

    id = Column(Integer, primary_key=True, index=True)
    scene_id = Column(Integer, ForeignKey("processed_scenes.id", ondelete="CASCADE"))
    label = Column(String, nullable=False, index=True)  # e.g., 'building', 'water_body'
    confidence = Column(Float, nullable=False)
    
    # Percentage coordinates for screen scaling [0, 100]
    x1 = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    x2 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)
    
    scene = relationship("ProcessedScene", back_populates="detections")


def init_db():
    """
    Initialize all database tables.
    """
    Base.metadata.create_all(bind=engine)

def get_db():
    """
    Dependency helper to retrieve database session inside API endpoints.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
