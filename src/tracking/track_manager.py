import time
import numpy as np
from src.tracking.motion_model import PlayerKalmanFilter
from src.tracking.lost_tracks import LostTracksPool
from src.tracking.assignment import AssignmentManager

class Track:
    """
    Representa un objeto seguido en tiempo real con su filtro de movimiento (Kalman)
    y memoria de características visuales acumulada con suavizado exponencial.
    """
    def __init__(self, track_id: int, class_id: int, real_pos: tuple, embedding: np.ndarray = None, color_hist: np.ndarray = None):
        self.id = track_id
        self.class_id = class_id
        self.state = 'candidate'  # Estados: 'candidate', 'active', 'lost'
        self.real_pos = real_pos
        self.last_known_pos = real_pos
        self.predicted_pos = real_pos
        
        # Filtro de Kalman con Aceleración Constante
        self.kalman = PlayerKalmanFilter(real_pos[0], real_pos[1])
        
        # Atributos visuales EMA
        self.embedding = embedding
        self.color_hist = color_hist
        
        self.hits = 1
        self.misses = 0
        self.lost_timestamp = 0.0

    def predict(self, dt: float = 0.04) -> tuple:
        """Avanza la predicción de movimiento de Kalman."""
        self.predicted_pos = self.kalman.predict(dt)
        return self.predicted_pos

    def update(self, real_pos: tuple, embedding: np.ndarray = None, color_hist: np.ndarray = None):
        """Actualiza el filtro de Kalman y las características visuales usando suavizado temporal (EMA)."""
        self.kalman.update(real_pos[0], real_pos[1])
        self.real_pos = self.kalman.get_pos()
        self.last_known_pos = self.real_pos
        self.hits += 1
        self.misses = 0
        
        # Actualizar embedding visual con EMA
        if embedding is not None:
            if self.embedding is not None:
                self.embedding = 0.9 * self.embedding + 0.1 * embedding
                norm = np.linalg.norm(self.embedding)
                if norm > 0:
                    self.embedding /= norm
            else:
                self.embedding = embedding
                
        # Actualizar histograma de color con EMA
        if color_hist is not None:
            if self.color_hist is not None:
                self.color_hist = 0.9 * self.color_hist + 0.1 * color_hist
            else:
                self.color_hist = color_hist

    def get_predicted_pos(self) -> tuple:
        return self.predicted_pos


class TrackManager:
    """
    Administra el ciclo de vida de los tracks. Valida apariciones sospechosas mediante
    el sistema de candidatos, y asocia detecciones huérfanas con la base de datos de ReID.
    """
    def __init__(self, max_lost_time: float = 6.0, min_hits_active: int = 5):
        self.min_hits_active = min_hits_active
        self.active_tracks = []
        self.candidate_tracks = []
        self.lost_pool = LostTracksPool(max_lost_time=max_lost_time)
        self.next_id = 1
        
    def _get_new_id(self) -> int:
        tid = self.next_id
        self.next_id += 1
        return tid

    def update(self, detections: list, dt: float = 0.04) -> list:
        # 1. Ejecutar la predicción de todos los tracks en seguimiento
        for track in self.active_tracks + self.candidate_tracks:
            track.predict(dt)
            
        # 2. Intentar asociar las detecciones actuales con los tracks en seguimiento (activos y candidatos)
        all_tracks = self.active_tracks + self.candidate_tracks
        cost_matrix = AssignmentManager.compute_cost_matrix(all_tracks, detections)
        matches, unmatched_tracks, unmatched_dets = AssignmentManager.match(cost_matrix, max_distance=0.7)
        
        matched_det_indices = set()
        for t_idx, d_idx in matches:
            track = all_tracks[t_idx]
            det = detections[d_idx]
            track.update(det["real_pos"], det.get("embedding"), det.get("color_hist"))
            matched_det_indices.add(d_idx)
            
            # Promoción de candidato a activo confirmado
            if track.state == 'candidate' and track.hits >= self.min_hits_active:
                track.state = 'active'
                if track in self.candidate_tracks:
                    self.candidate_tracks.remove(track)
                    self.active_tracks.append(track)
                    
        # 3. Mover los tracks no actualizados a su estado correspondiente
        for idx in unmatched_tracks:
            track = all_tracks[idx]
            track.misses += 1
            
            if track.state == 'active':
                # Si era activo, se pasa al pool de perdidos para posible ReID posterior
                track.state = 'lost'
                self.active_tracks.remove(track)
                self.lost_pool.add(track)
            elif track.state == 'candidate':
                # Los candidatos con oclusión o ruido se eliminan de inmediato para evitar falsos positivos
                if track in self.candidate_tracks:
                    self.candidate_tracks.remove(track)

        # 4. Limpiar tracks viejos del pool de oclusión expirados
        self.lost_pool.clean_old_tracks()

        # 5. RE-IDENTIFICACIÓN (ReID): Asociar detecciones huérfanas con tracks del pool de perdidos
        unmatched_det_list = [detections[idx] for idx in unmatched_dets]
        lost_tracks_list = list(self.lost_pool.pool.values())
        
        if len(lost_tracks_list) > 0 and len(unmatched_det_list) > 0:
            reid_cost_matrix = AssignmentManager.compute_cost_matrix(lost_tracks_list, unmatched_det_list)
            # ReID es más estricto con los límites de costo
            reid_matches, reid_unmatched_tracks, reid_unmatched_dets = AssignmentManager.match(reid_cost_matrix, max_distance=0.55)
            
            for t_idx, d_idx in reid_matches:
                track = lost_tracks_list[t_idx]
                det = unmatched_det_list[d_idx]
                
                # Revivir identidad persistente del jugador
                track.state = 'active'
                track.update(det["real_pos"], det.get("embedding"), det.get("color_hist"))
                
                self.lost_pool.pop(track.id)
                self.active_tracks.append(track)
                
                # Registrar que esta detección ya fue asociada
                original_det_idx = unmatched_dets[d_idx]
                matched_det_indices.add(original_det_idx)
                
        # 6. Crear un nuevo track en modo CANDIDATO para las detecciones huérfanas restantes
        for idx, det in enumerate(detections):
            if idx not in matched_det_indices:
                new_track = Track(
                    track_id=self._get_new_id(),
                    class_id=det["class_id"],
                    real_pos=det["real_pos"],
                    embedding=det.get("embedding"),
                    color_hist=det.get("color_hist")
                )
                self.candidate_tracks.append(new_track)
                
        return self.active_tracks
