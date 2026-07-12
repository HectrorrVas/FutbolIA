import numpy as np

class PlayerKalmanFilter:
    """
    Filtro de Kalman con modelo de Aceleración Constante (CA) en coordenadas reales 2D (metros).
    Esto modela mejor los cambios repentinos de velocidad de los jugadores.
    """
    def __init__(self, x0: float, y0: float, dt: float = 0.04):
        self.dt = dt
        # Estado: [x, y, vx, vy, ax, ay]^T
        self.x = np.array([x0, y0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Matriz de transición de estado F
        self.F = np.array([
            [1.0, 0.0, dt,  0.0, 0.5 * dt**2, 0.0],
            [0.0, 1.0, 0.0, dt,  0.0,         0.5 * dt**2],
            [0.0, 0.0, 1.0, 0.0, dt,          0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0,         dt],
            [0.0, 0.0, 0.0, 0.0, 1.0,         0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0,         1.0]
        ], dtype=np.float32)
        
        # Matriz de medición H (medimos posición X, Y)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        ], dtype=np.float32)
        
        # Covarianza del ruido del proceso Q
        self.Q = np.eye(6, dtype=np.float32) * 0.15
        
        # Covarianza del ruido de medición R (metros reales)
        self.R = np.eye(2, dtype=np.float32) * 0.30
        
        # Covarianza de error P
        self.P = np.eye(6, dtype=np.float32) * 2.0

    def predict(self, dt: float = None) -> tuple:
        """
        Avanza el estado del filtro de Kalman al siguiente paso de tiempo.
        """
        if dt is not None and dt != self.dt:
            self.dt = dt
            self.F[0, 2] = dt
            self.F[1, 3] = dt
            self.F[0, 4] = 0.5 * dt**2
            self.F[1, 5] = 0.5 * dt**2
            self.F[2, 4] = dt
            self.F[3, 5] = dt
            
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return float(self.x[0]), float(self.x[1])

    def update(self, z_x: float, z_y: float):
        """
        Actualiza el estado con una nueva medición en metros reales.
        """
        z = np.array([z_x, z_y], dtype=np.float32)
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

    def get_pos(self) -> tuple:
        return float(self.x[0]), float(self.x[1])

    def get_velocity(self) -> tuple:
        return float(self.x[2]), float(self.x[3])
