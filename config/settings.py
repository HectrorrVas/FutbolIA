from pathlib import Path
import sys

# Detectar automáticamente si estamos en Google Colab o en local
if 'google.colab' in sys.modules:
    PROJECT_DIR = Path("/content/Futbol2026")
else:
    PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "model" / "best.pt"
VIDEO_PATH = PROJECT_DIR / "video" / "video1.mp4"

OUTPUT_DIR = PROJECT_DIR / "output"

CLIPS_DIR = PROJECT_DIR / "clips"
ORIGINAL_CLIPS_DIR = CLIPS_DIR / "original"
PROCESSED_CLIPS_DIR = CLIPS_DIR / "processed"

LOGS_DIR = PROJECT_DIR / "logs"

# =====================================================
# CONFIGURACIÓN DEL VIDEO
# =====================================================

START_MINUTE = 0
END_MINUTE = 1

OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
OUTPUT_FPS = 10

# =====================================================
# CONFIGURACIÓN DE YOLO
# =====================================================

IMAGE_SIZE = 640
CONFIDENCE = 0.25
FRAME_SKIP = 1