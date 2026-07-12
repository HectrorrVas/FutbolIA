import time

class LostTracksPool:
    """
    Pool de almacenamiento y gestión de tracks perdidos temporalmente.
    Permite recuperar la identidad (ReID) cuando un jugador vuelve a aparecer.
    """
    def __init__(self, max_lost_time: float = 6.0):
        self.max_lost_time = max_lost_time
        self.pool = {}  # {track_id: Track}

    def add(self, track):
        """Agrega un track al pool registrando el timestamp del momento en que se perdió."""
        track.lost_timestamp = time.time()
        self.pool[track.id] = track

    def pop(self, track_id):
        """Retira y retorna un track del pool al ser re-identificado."""
        return self.pool.pop(track_id, None)

    def clean_old_tracks(self):
        """Elimina permanentemente los tracks que han superado el tiempo máximo de oclusión."""
        now = time.time()
        expired = [
            tid for tid, track in self.pool.items()
            if now - track.lost_timestamp > self.max_lost_time
        ]
        for tid in expired:
            del self.pool[tid]

    @staticmethod
    def check_border_reentry_affinity(track_pos: tuple, det_pos: tuple, threshold: float = 2.5) -> float:
        """
        Calcula un bonus de afinidad (0.0 a 1.0) si un jugador desaparecido cerca de un borde
        reaparece cerca del mismo borde (cámara de dron perdiendo y recuperando jugadores por las bandas).
        """
        tx, ty = track_pos
        dx, dy = det_pos
        
        # Borde izquierdo (X ~ 0)
        if tx < threshold and dx < threshold:
            return 1.0
        # Borde derecho (X ~ 41)
        if tx > 41.0 - threshold and dx > 41.0 - threshold:
            return 1.0
        # Borde inferior (Y ~ 0)
        if ty < threshold and dy < threshold:
            return 1.0
        # Borde superior (Y ~ 68)
        if ty > 68.0 - threshold and dy > 68.0 - threshold:
            return 1.0
            
        return 0.0
