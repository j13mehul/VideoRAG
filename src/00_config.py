# VideoRAG Configuration
import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "sampledata"  # Using sample dataset for processing
SAMPLE_DATA_DIR = PROJECT_ROOT / "sampledata"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
UPLOADED_OUTPUT_DIR = PROJECT_ROOT / "uploaded_output"
CACHE_DIR = PROJECT_ROOT / "cache"

# Video processing settings
CHUNK_DURATION = 3  # seconds
KEYFRAMES_PER_CHUNK = 1  # 1-3 keyframes per chunk
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']

# Model settings
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
quick_gelu = True
YOLO_MODEL = "yolov8n.pt"  # Use nano model for speed

# FAISS settings
FAISS_INDEX_PATH = OUTPUT_DIR / "faiss_index.bin"
METADATA_PATH = OUTPUT_DIR / "metadata.jsonl"
EMBEDDING_DIM = 512  # CLIP ViT-B-32 embedding dimension

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
LLAVA_MODEL = "llava:7b"

# Query settings
TOP_K_RETRIEVAL = 5
CONFIDENCE_THRESHOLD = 0.15  # Lowered for smaller sample dataset

# Create directories if they don't exist
for directory in [OUTPUT_DIR, CACHE_DIR, UPLOADED_OUTPUT_DIR]:
    directory.mkdir(exist_ok=True, parents=True)