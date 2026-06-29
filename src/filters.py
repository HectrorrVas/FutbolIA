import math
import numpy as np

class LowPassFilter:
    """Filtro de paso bajo simple para suavizado."""
    def __init__(self, alpha):
        self.alpha = alpha
        self.y = None

    def filter(self, value):
        if self.y is None:
            self.y = value
        else:
            self.y = self.alpha * value + (1.0 - self.alpha) * self.y
        return self.y


class OneEuroFilter:
    """Implementacion de One Euro Filter para suavizado adaptativo en 1D."""
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filt = LowPassFilter(self.alpha(min_cutoff))
        self.dx_filt = LowPassFilter(self.alpha(d_cutoff))
        self.t_prev = t0
        self.x_prev = x0

    def alpha(self, cutoff, dt=1.0):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, t, x):
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_filt.y if self.x_filt.y is not None else x

        # Estimacion de la derivada
        dx = (x - self.x_prev) / dt
        
        # Filtrar derivada
        alpha_d = self.alpha(self.d_cutoff, dt)
        self.dx_filt.alpha = alpha_d
        edx = self.dx_filt.filter(dx)
        
        # Calcular frecuencia de corte adaptativa
        cutoff = self.min_cutoff + self.beta * abs(edx)
        
        # Filtrar valor
        alpha = self.alpha(cutoff, dt)
        self.x_filt.alpha = alpha
        x_filtered = self.x_filt.filter(x)
        
        # Guardar estado
        self.t_prev = t
        self.x_prev = x
        return x_filtered


class OneEuroFilter2D:
    """One Euro Filter para coordenadas 2D (x, y)."""
    def __init__(self, t0, x0, y0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.x_filter = OneEuroFilter(t0, x0, min_cutoff, beta, d_cutoff)
        self.y_filter = OneEuroFilter(t0, y0, min_cutoff, beta, d_cutoff)

    def filter(self, t, x, y):
        x_filt = self.x_filter.filter(t, x)
        y_filt = self.y_filter.filter(t, y)
        return x_filt, y_filt


class BallKalmanFilter:
    """Filtro de Kalman para el balon con modelo de velocidad constante."""
    def __init__(self, x0, y0, dt=0.1):
        self.dt = dt
        # Estado: [x, y, vx, vy]^T
        self.x = np.array([x0, y0, 0.0, 0.0], dtype=np.float32)
        
        # Matriz de transicion de estado
        self.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Matriz de medicion
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float32)
        
        # Covarianza del ruido del proceso (Q)
        self.Q = np.eye(4, dtype=np.float32) * 0.05
        
        # Covarianza del ruido de la medicion (R)
        self.R = np.eye(2, dtype=np.float32) * 0.5
        
        # Covarianza del error (P)
        self.P = np.eye(4, dtype=np.float32) * 5.0
        
        # Contador de frames sin detecciones
        self.missed_frames = 0

    def predict(self, dt=None):
        if dt is not None and dt != self.dt:
            self.dt = dt
            self.F[0, 2] = dt
            self.F[1, 3] = dt
        
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return float(self.x[0]), float(self.x[1])

    def update(self, z_x, z_y):
        z = np.array([z_x, z_y], dtype=np.float32)
        y = z - np.dot(self.H, self.x)  # Residuo de medicion
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R  # Covarianza del residuo
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Ganancia de Kalman
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        self.missed_frames = 0

    def get_pos(self):
        return float(self.x[0]), float(self.x[1])
