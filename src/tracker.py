import numpy as np
from ultralytics import YOLO

class ByteTrackManager:
    """Clase encargada de gestionar la inferencia de YOLO con tracking ByteTrack."""
    def __init__(self, model_path, conf=0.25, imgsz=640):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz

    def track(self, frame):
        """
        Ejecuta el tracking en el frame dado.
        Retorna los resultados del tracking.
        """
        # Se fuerza el uso de ByteTrack mediante tracker="bytetrack.yaml"
        results = self.model.track(
            source=frame,
            persist=True,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
            tracker="bytetrack.yaml"
        )
        return results[0] if results else None
