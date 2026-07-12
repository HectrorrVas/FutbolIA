import numpy as np
from src.tracking.lost_tracks import LostTracksPool

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

class AssignmentManager:
    """
    Gestor de asociación de detecciones a tracks mediante métricas multimodales
    y el algoritmo Húngaro para optimización global del costo de emparejamiento.
    """
    @staticmethod
    def compute_cost_matrix(tracks: list, detections: list, border_threshold: float = 2.5) -> np.ndarray:
        n_tracks = len(tracks)
        n_detections = len(detections)
        if n_tracks == 0 or n_detections == 0:
            return np.empty((n_tracks, n_detections))
            
        cost_matrix = np.zeros((n_tracks, n_detections), dtype=np.float32)
        
        for i, track in enumerate(tracks):
            track_pos = track.get_predicted_pos()
            track_class = track.class_id
            track_emb = track.embedding
            track_hist = track.color_hist
            
            for j, det in enumerate(detections):
                det_pos = det["real_pos"]
                det_class = det["class_id"]
                det_emb = det.get("embedding", None)
                det_hist = det.get("color_hist", None)
                
                # 1. Restricción estricta de clase (evitar emparejar porteros con árbitros, etc.)
                if track_class != det_class:
                    cost_matrix[i, j] = 1.0  # Costo máximo absoluto
                    continue
                
                # 2. Distancia física real sobre la cancha (en metros)
                dist = np.linalg.norm(np.array(track_pos) - np.array(det_pos))
                
                # Si está físicamente a más de 12 metros de su posición predicha (físicamente improbable), penalizar
                if dist > 12.0:
                    cost_pos = 1.0
                else:
                    cost_pos = dist / 12.0
                
                # 3. Similitud visual (Deep embedding cosine distance)
                cost_visual = 0.5
                if track_emb is not None and det_emb is not None:
                    # Distancia coseno mapeada a [0, 1]
                    cosine_dist = 1.0 - np.dot(track_emb, det_emb)
                    cost_visual = np.clip(cosine_dist, 0.0, 1.0)
                
                # 4. Similitud de color (Histograma HSV)
                cost_color = 0.5
                if track_hist is not None and det_hist is not None:
                    # Suma de diferencias absolutas (L1 normalizado)
                    diff = np.abs(track_hist - det_hist).sum()
                    cost_color = np.clip(diff / 2.0, 0.0, 1.0)
                
                # 5. Bonus de reingreso por borde (reduce el costo del emparejamiento)
                bonus_reentry = 0.0
                if getattr(track, "state", None) == 'lost':
                    # Si el track está perdido, validar si reingresó por el mismo borde
                    bonus_reentry = 0.35 * LostTracksPool.check_border_reentry_affinity(
                        track.last_known_pos, det_pos, border_threshold
                    )
                
                # Combinar costos con pesos ponderados
                cost = 0.4 * cost_pos + 0.3 * cost_visual + 0.3 * cost_color - bonus_reentry
                cost_matrix[i, j] = np.clip(cost, 0.0, 1.0)
                
        return cost_matrix

    @staticmethod
    def match(cost_matrix: np.ndarray, max_distance: float = 0.65) -> tuple:
        """
        Resuelve la asignación lineal de mínimo costo utilizando el Algoritmo Húngaro.
        Si scipy no está disponible, cae de manera segura a un algoritmo codicioso (greedy).
        """
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
            
        matches = []
        unmatched_tracks = []
        unmatched_detections = []
        
        if HAS_SCIPY:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < max_distance:
                    matches.append((r, c))
                else:
                    unmatched_tracks.append(r)
                    unmatched_detections.append(c)
            # Agregar filas y columnas sobrantes no asignadas por el algoritmo
            for r in range(cost_matrix.shape[0]):
                if r not in row_ind:
                    unmatched_tracks.append(r)
            for c in range(cost_matrix.shape[1]):
                if c not in col_ind:
                    unmatched_detections.append(c)
        else:
            # Fallback Greedy (Búsqueda codiciosa del mínimo elemento)
            matrix = cost_matrix.copy()
            while True:
                min_idx = np.unravel_index(np.argmin(matrix), matrix.shape)
                min_val = matrix[min_idx]
                if min_val >= max_distance or min_val == 1.0:
                    break
                r, c = min_idx
                matches.append((r, c))
                matrix[r, :] = 1.0
                matrix[:, c] = 1.0
                
            for r in range(cost_matrix.shape[0]):
                if not any(m[0] == r for m in matches):
                    unmatched_tracks.append(r)
            for c in range(cost_matrix.shape[1]):
                if not any(m[1] == c for m in matches):
                    unmatched_detections.append(c)
                    
        return matches, unmatched_tracks, unmatched_detections
