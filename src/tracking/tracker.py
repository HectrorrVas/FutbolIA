import cv2
import numpy as np
from src.tracking.appearance import AppearanceModel
from src.tracking.track_manager import TrackManager
from src.tracking.filters import TrackingFilters

class FootballTracker:
    """
    Orquestador principal del sistema de Tracking y Re-Identificación (ReID) para FutbolIA.
    Transforma coordenadas de la imagen a metros reales utilizando la homografía calculada,
    y filtra elementos fuera de cancha antes de realizar el emparejamiento.
    """
    def __init__(self, min_player_conf: float = 0.25):
        self.min_player_conf = min_player_conf
        self.appearance_model = AppearanceModel()
        self.track_manager = TrackManager(max_lost_time=6.0, min_hits_active=5)
        self.last_timestamp = None

    def update(self, frame: np.ndarray, detection_results, estimator, timestamp: float) -> list:
        """
        Procesa el frame actual y las detecciones crudas de YOLO para retornar la lista de
        tracks activos identificados persistentes.
        """
        detections = []
        h, w = frame.shape[:2]
        
        # 1. Calcular paso de tiempo dt
        dt = 0.04
        if self.last_timestamp is not None:
            dt = max(0.001, timestamp - self.last_timestamp)
        self.last_timestamp = timestamp

        if detection_results is not None and len(detection_results.boxes) > 0:
            boxes = detection_results.boxes
            cls_arr = boxes.cls.cpu().numpy().astype(int)
            xyxy_arr = boxes.xyxy.cpu().numpy()
            conf_arr = boxes.conf.cpu().numpy()
            
            for i in range(len(boxes)):
                cls_id = cls_arr[i]
                conf = conf_arr[i]
                
                # Filtrar clases: 0 (árbitro), 2 (equipo A), 3 (equipo B), 5 (portero)
                # Ignorar clase 1 (balón), que tiene su propia lógica de filtrado y suavizado.
                if cls_id not in [0, 2, 3, 5] or conf < self.min_player_conf:
                    continue
                    
                x1, y1, x2, y2 = xyxy_arr[i]
                
                # Posición de los pies: centro horizontal y extremo inferior de la caja de detección
                cx = (x1 + x2) / 2.0
                cy = y2
                
                # 2. Proyectar coordenadas de píxeles a metros reales de la cancha
                if estimator is not None and estimator.H is not None:
                    try:
                        real_pos = estimator.transform_image_to_real([(cx, cy)])[0]
                    except Exception:
                        continue
                else:
                    # Sin homografía disponible no se puede procesar el track
                    continue
                    
                # 3. Filtrar detecciones espureas fuera del terreno de juego (espectadores, suplentes, etc.)
                if not TrackingFilters.is_inside_field(real_pos[0], real_pos[1]):
                    continue
                    
                # 4. Extraer embeddings profundos y características de color HSV (crop de la caja de detección)
                ix1, iy1 = max(0, int(x1)), max(0, int(y1))
                ix2, iy2 = min(w - 1, int(x2)), min(h - 1, int(y2))
                crop = frame[iy1:iy2, ix1:ix2]
                
                embedding = self.appearance_model.get_deep_embedding(crop)
                color_hist = self.appearance_model.get_color_histogram(crop)
                
                detections.append({
                    "class_id": cls_id,
                    "img_pos": (cx, cy),
                    "real_pos": (real_pos[0], real_pos[1]),
                    "bbox": (x1, y1, x2, y2),
                    "embedding": embedding,
                    "color_hist": color_hist
                })
                
        # 5. Actualizar el gestor de tracks con las detecciones refinadas
        active_tracks = self.track_manager.update(detections, dt)
        
        # 6. Formatear salida estructurada para renderizado y análisis
        formatted_tracks = []
        for t in active_tracks:
            # Transformar la posición física suavizada (Kalman) de vuelta a píxeles de imagen
            try:
                img_pos_arr = estimator.transform_real_to_image([t.real_pos])[0]
                img_pos = (float(img_pos_arr[0]), float(img_pos_arr[1]))
            except Exception:
                img_pos = t.last_known_pos  # fallback si falla la proyección inversa
                
            formatted_tracks.append({
                "track_id": t.id,
                "class_id": t.class_id,
                "center": img_pos,       # Coordenadas en píxeles del frame
                "real_pos": t.real_pos   # Coordenadas en metros reales (X, Y)
            })
            
        return formatted_tracks
