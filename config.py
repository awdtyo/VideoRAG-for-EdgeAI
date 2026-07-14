"""
Central configuration for the local video RAG pipeline.
Tune SAMPLE_FPS / thresholds down further if you hit RAM/CPU limits on the Pi.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "weights")
DB_DIR = os.path.join(BASE_DIR, "data", "db")
FRAME_DIR = os.path.join(BASE_DIR, "data", "frames")
AUDIO_DIR = os.path.join(BASE_DIR, "data", "audio")

for d in (MODEL_DIR, DB_DIR, FRAME_DIR, AUDIO_DIR):
    os.makedirs(d, exist_ok=True)

# ---- Ingestion ----
SAMPLE_FPS = 1.0          # frames sampled per second of video (lower = less RAM/CPU)
MAX_FRAME_DIM = 640        # frames are resized so the longer side = this, before detection

# ---- Detection (YOLOX-Nano ONNX) ----
DETECTOR_ONNX_PATH = os.path.join(MODEL_DIR, "yolox_nano.onnx")
DETECTOR_INPUT_SIZE = (416, 416)   # YOLOX-Nano default
DETECTOR_CONF_THRESH = 0.35
DETECTOR_NMS_THRESH = 0.45
COCO_CLASSES_PATH = os.path.join(BASE_DIR, "coco_classes.txt")

# ---- Frame embedding (MobileNetV3-Small ONNX) ----
EMBEDDER_ONNX_PATH = os.path.join(MODEL_DIR, "mobilenetv3_small.onnx")
EMBEDDER_INPUT_SIZE = (224, 224)
ENABLE_VISUAL_EMBEDDINGS = True   # set False to skip (saves RAM/CPU, text-only RAG still works)

# ---- OCR ----
ENABLE_OCR = True
OCR_EVERY_N_FRAMES = 3     # only run tesseract every Nth sampled frame

# ---- Audio transcription (faster-whisper) ----
WHISPER_MODEL_SIZE = "tiny"     # "tiny" or "base" recommended on Pi 5 4GB
WHISPER_COMPUTE_TYPE = "int8"    # int8 quantized = much lower RAM

# ---- Text embedding + vector store ----
TEXT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # ~130MB, good quality/size tradeoff
CHROMA_COLLECTION = "video_rag"

# ---- Local LLM generation (via Ollama) ----
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:3b"     # pull with: ollama pull qwen2.5:3b
TOP_K_RETRIEVE = 6
