import sys
from pathlib import Path

# Agregar el directorio raíz del proyecto al path del sistema
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import (
    MODEL_PATH,
    ORIGINAL_CLIPS_DIR,
    PROCESSED_CLIPS_DIR,
    CONFIDENCE,
    IMAGE_SIZE,
)

from src.processor import VideoProcessor

def main():
    print("=" * 80)
    print("SISTEMA DE TRACKING Y SUAVIZADO PROFESIONAL PARA ANALISIS DE FUTBOL")
    print("=" * 80)

    # Crear directorios si no existen
    PROCESSED_CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    input_path = ORIGINAL_CLIPS_DIR / "clip_01.mp4"
    output_path = PROCESSED_CLIPS_DIR / "clip_01_processed.mp4"

    if not input_path.exists():
        print(f"[ERROR] No se encontro el archivo de video de entrada: {input_path}")
        print("Asegurate de colocar el clip en la ruta correspondiente.")
        sys.exit(1)

    # Inicializar el procesador de video
    processor = VideoProcessor(
        model_path=str(MODEL_PATH),
        confidence=CONFIDENCE,
        imgsz=IMAGE_SIZE
    )

    # Ejecutar pipeline
    try:
        processor.process(
            input_video_path=input_path,
            output_video_path=output_path
        )
        print("\n[OK] Pipeline finalizado con exito.")
    except Exception as e:
        print(f"\n[ERROR] Ocurrio un problema durante el procesamiento: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
