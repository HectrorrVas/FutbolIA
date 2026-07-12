import cv2
import numpy as np
from typing import Dict, Tuple, List, Union, Optional

# =====================================================================
# MODELO GEOMÉTRICO DE LA CANCHA VERTICAL (Fútbol 7/9 - 41m x 68m)
# =====================================================================
# Coordenadas reales (X, Y) en metros de los 20 puntos clave.
# El origen (0, 0) está en el córner inferior izquierdo (KP12).
# Eje X: 0 a 41 metros (ancho, horizontal en el dibujo).
# Eje Y: 0 a 68 metros (largo, vertical en el dibujo).
FIELD_POINTS: Dict[str, Tuple[float, float]] = {
    "KP12": (0.0, 68.0),     # Córner superior izquierdo (rotado sobre Y)
    "KP13": (7.0, 68.0),     # Área grande superior - intersección izquierda (rotado sobre Y)
    "KP14": (34.0, 68.0),    # Área grande superior - intersección derecha (rotado sobre Y)
    "KP15": (41.0, 68.0),    # Córner superior derecho (rotado sobre Y)
    
    "KP08": (7.0, 57.0),     # Esquina inferior izquierda del área grande superior (rotado sobre Y)
    "KP09": (16.0, 57.0),    # Intersección inferior de la medialuna superior con área (rotado sobre Y)
    "KP10": (25.0, 57.0),    # Intersección superior de la medialuna superior con área (rotado sobre Y)
    "KP11": (34.0, 57.0),    # Esquina inferior derecha del área grande superior (rotado sobre Y)
    
    "KP00": (0.0, 34.0),     # Centro de la línea de banda izquierda (rotado sobre Y)
    "KP01": (15.5, 34.0),    # Intersección izquierda del círculo central (rotado sobre Y)
    "KP02": (25.5, 34.0),    # Intersección derecha del círculo central (rotado sobre Y)
    "KP03": (41.0, 34.0),    # Centro de la línea de banda derecha (rotado sobre Y)
    
    "KP04": (7.0, 11.0),     # Esquina superior izquierda del área grande inferior (rotado sobre Y)
    "KP05": (16.0, 11.0),    # Intersección superior de la medialuna inferior con área (rotado sobre Y)
    "KP06": (25.0, 11.0),    # Intersección inferior de la medialuna inferior con área (rotado sobre Y)
    "KP07": (34.0, 11.0),    # Esquina superior derecha del área grande inferior (rotado sobre Y)
    
    "KP16": (0.0, 0.0),      # Córner inferior izquierdo (rotado sobre Y)
    "KP17": (7.0, 0.0),      # Área grande inferior - intersección izquierda (rotado sobre Y)
    "KP18": (34.0, 0.0),     # Área grande inferior - intersección derecha (rotado sobre Y)
    "KP19": (41.0, 0.0),     # Córner inferior derecho (rotado sobre Y)
}


def meters_to_px(x_m: float, y_m: float, scale: float = 10.0, padding: int = 25) -> Tuple[int, int]:
    """
    Convierte coordenadas reales en metros (X: ancho, Y: largo) a píxeles de OpenCV,
    respetando la orientación vertical donde Y=0 es abajo e Y=68 es arriba.
    
    Args:
        x_m: Coordenada X real (0 a 41 metros).
        y_m: Coordenada Y real (0 a 68 metros).
        scale: Factor de escala (píxeles por metro).
        padding: Margen exterior en píxeles.
        
    Returns:
        Tuple[int, int]: (px, py) en píxeles de la imagen.
    """
    h_px = int(68.0 * scale) + 2 * padding
    px = int(x_m * scale) + padding
    py = h_px - (int(y_m * scale) + padding)
    return px, py


class HomographyEstimator:
    """
    Clase encargada de calcular y gestionar la matriz de homografía entre el video
    y el plano 2D real de la cancha vertical en metros.
    """
    def __init__(self, min_confidence: float = 0.3):
        self.min_confidence = min_confidence
        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None
        
        # Guardar qué puntos fueron inliers/outliers en la última estimación
        self.last_inliers: Dict[str, bool] = {}
        self.last_detected: Dict[str, bool] = {}

    def estimate(self, detected_kps: Union[Dict[str, Tuple[float, float, float]], np.ndarray]) -> Tuple[bool, int]:
        """
        Calcula la matriz de homografía (H) usando RANSAC o transformación afín.
        Soporta estimación a partir de un mínimo de 3 puntos detectados.
        Aplica suavizado temporal (EMA) con la homografía anterior para evitar saltos bruscos.
        
        Args:
            detected_kps: Puede ser:
                - Un diccionario: {"KP00": (x, y, conf), ...}
                - Un array numpy de forma (20, 3) o (20, 2) en el orden de KP00 a KP19.
                  Si es (20, 2), se asume confianza = 1.0 para todos.
                  
        Returns:
            Tuple[bool, int]: (True si se calculó con éxito o se mantiene la previa, número de inliers)
        """
        src_pts = []  # Coordenadas en la imagen (x, y)
        dst_pts = []  # Coordenadas reales (X, Y)
        kp_names_ordered = []

        # Resetear estados de detección e inliers para el frame actual
        self.last_inliers = {}
        self.last_detected = {}

        # 1. Parsear y filtrar los puntos según confianza y presencia
        if isinstance(detected_kps, dict):
            for name, real_pos in FIELD_POINTS.items():
                if name in detected_kps:
                    x, y, conf = detected_kps[name]
                    self.last_detected[name] = True
                    if conf >= self.min_confidence:
                        src_pts.append([x, y])
                        dst_pts.append(list(real_pos))
                        kp_names_ordered.append(name)
                    else:
                        self.last_inliers[name] = False
                else:
                    self.last_detected[name] = False
                    self.last_inliers[name] = False
        else:
            # Asumir array numpy de forma (20, 2) o (20, 3) en el orden correlativo de KP00 a KP19
            kp_names = [f"KP{i:02d}" for i in range(20)]
            for i, name in enumerate(kp_names):
                if i < len(detected_kps):
                    kp_data = detected_kps[i]
                    x, y = kp_data[0], kp_data[1]
                    conf = kp_data[2] if len(kp_data) > 2 else 1.0
                    
                    self.last_detected[name] = True
                    if conf >= self.min_confidence:
                        src_pts.append([x, y])
                        dst_pts.append(list(FIELD_POINTS[name]))
                        kp_names_ordered.append(name)
                    else:
                        self.last_inliers[name] = False
                else:
                    self.last_detected[name] = False
                    self.last_inliers[name] = False

        # 2. Se requieren al menos 3 puntos para estimar la homografía
        if len(src_pts) < 3:
            if self.H is not None:
                # Retornar True usando la matriz del frame anterior
                return True, 0
            return False, 0

        src_pts_arr = np.array(src_pts, dtype=np.float32)
        dst_pts_arr = np.array(dst_pts, dtype=np.float32)

        H_actual = None
        num_inliers = 0

        # 3. Calcular homografía (o transformación afín si hay exactamente 3 puntos)
        if len(src_pts) == 3:
            A, inliers_status = cv2.estimateAffine2D(src_pts_arr, dst_pts_arr)
            if A is not None:
                H_actual = np.eye(3)
                H_actual[:2, :] = A
                # Marcar los 3 puntos como inliers
                for name in kp_names_ordered:
                    self.last_inliers[name] = True
                num_inliers = 3
        else:
            # RANSAC con 4 o más puntos
            H_res, status = cv2.findHomography(src_pts_arr, dst_pts_arr, cv2.RANSAC, 5.0)
            if H_res is not None:
                H_actual = H_res
                status_flat = status.flatten()
                for idx, name in enumerate(kp_names_ordered):
                    is_inlier = bool(status_flat[idx] == 1)
                    self.last_inliers[name] = is_inlier
                    if is_inlier:
                        num_inliers += 1

        # 4. Aplicar suavizado temporal (EMA): H = 0.9 * H_anterior + 0.1 * H_actual
        if H_actual is not None:
            # Normalizar matriz actual para la consistencia de escala
            if H_actual[2, 2] != 0:
                H_actual = H_actual / H_actual[2, 2]
                
            if self.H is not None:
                self.H = 0.9 * self.H + 0.1 * H_actual
                if self.H[2, 2] != 0:
                    self.H = self.H / self.H[2, 2]
            else:
                self.H = H_actual
                
            _, self.H_inv = cv2.invert(self.H)
            return True, num_inliers
        else:
            # Si no se pudo calcular la matriz en este frame pero hay una previa, se preserva
            if self.H is not None:
                return True, 0
            return False, 0

    def transform_image_to_real(self, points: Union[List[Tuple[float, float]], np.ndarray]) -> np.ndarray:
        """
        Transforma puntos de coordenadas de la imagen (píxeles) a coordenadas de la cancha (metros).
        
        Args:
            points: Lista o array de puntos [(x1, y1), (x2, y2), ...]
            
        Returns:
            np.ndarray: Array de puntos transformados [(X1, Y1), (X2, Y2), ...]
        """
        if self.H is None:
            raise ValueError("La matriz de homografía no ha sido calculada con éxito.")

        pts = np.array(points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 1, 2)
        elif pts.ndim == 2:
            pts = pts.reshape(-1, 1, 2)

        transformed = cv2.perspectiveTransform(pts, self.H)
        return transformed.reshape(-1, 2)

    def transform_real_to_image(self, points: Union[List[Tuple[float, float]], np.ndarray]) -> np.ndarray:
        """
        Transforma puntos de coordenadas de la cancha (metros) a coordenadas de la imagen (píxeles).
        
        Args:
            points: Lista o array de puntos [(X1, Y1), (X2, Y2), ...]
            
        Returns:
            np.ndarray: Array de puntos transformados [(x1, y1), (x2, y2), ...]
        """
        if self.H_inv is None:
            raise ValueError("La matriz de homografía inversa no está disponible.")

        pts = np.array(points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 1, 2)
        elif pts.ndim == 2:
            pts = pts.reshape(-1, 1, 2)

        transformed = cv2.perspectiveTransform(pts, self.H_inv)
        return transformed.reshape(-1, 2)


# =====================================================================
# RENDERIZADO Y DIBUJO DE LA CANCHA 2D VERTICAL
# =====================================================================

def draw_pitch_2d(scale: float = 10.0, padding: int = 25) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Dibuja una representación 2D cenital y limpia de la cancha en posición vertical (41m x 68m).
    
    Args:
        scale: Factor de escala (píxeles por metro). Por defecto 10 píxeles = 1 metro.
        padding: Margen exterior en píxeles.
        
    Returns:
        Tuple[np.ndarray, Tuple[int, int]]: (Imagen de la cancha, (ancho_total, alto_total))
    """
    # Dimensiones en metros de la cancha vertical
    field_w_m = 41.0
    field_h_m = 68.0
    
    # Dimensiones de la imagen resultante en píxeles
    w_px = int(field_w_m * scale) + 2 * padding
    h_px = int(field_h_m * scale) + 2 * padding
    
    # Crear canvas verde premium (BGR)
    img = np.zeros((h_px, w_px, 3), dtype=np.uint8)
    img[:] = (35, 145, 60)  # Verde campo
    
    c_white = (255, 255, 255)
    lw = 2  # Grosor de línea
    
    # Función local para acortar llamadas
    def to_px(x_m: float, y_m: float) -> Tuple[int, int]:
        return meters_to_px(x_m, y_m, scale, padding)

    # 1. Borde exterior del campo
    cv2.rectangle(img, to_px(0.0, 0.0), to_px(41.0, 68.0), c_white, lw, lineType=cv2.LINE_AA)
    
    # 2. Línea central
    cv2.line(img, to_px(0.0, 34.0), to_px(41.0, 34.0), c_white, lw, lineType=cv2.LINE_AA)
    
    # 3. Círculo central y punto de saque
    r_center = int(5.0 * scale)
    cv2.circle(img, to_px(20.5, 34.0), r_center, c_white, lw, lineType=cv2.LINE_AA)
    cv2.circle(img, to_px(20.5, 34.0), 3, c_white, -1, lineType=cv2.LINE_AA)
    
    # 4. Área penal inferior (Grande)
    cv2.rectangle(img, to_px(7.0, 0.0), to_px(34.0, 11.0), c_white, lw, lineType=cv2.LINE_AA)
    
    # Área chica inferior: 13m ancho (7m portería + 3m c/lado) × 5.5m profundidad
    # Estimado por proporción: ~50% de los 11m del área grande (reglamento fútbol 7/9)
    cv2.rectangle(img, to_px(14.0, 0.0), to_px(27.0, 5.5), c_white, lw, lineType=cv2.LINE_AA)
    
    # Medialuna inferior (Arco de área)
    center_l = to_px(20.5, 9.0)
    cv2.circle(img, center_l, 2, c_white, -1, lineType=cv2.LINE_AA)  # Punto de penal inferior
    # Ángulo centrado hacia arriba (270 grados en OpenCV es arriba)
    cv2.ellipse(img, center_l, (r_center, r_center), 0, 217, 323, c_white, lw, lineType=cv2.LINE_AA)

    # 5. Área penal superior (Grande)
    cv2.rectangle(img, to_px(7.0, 57.0), to_px(34.0, 68.0), c_white, lw, lineType=cv2.LINE_AA)
    
    # Área chica superior: 13m ancho (7m portería + 3m c/lado) × 5.5m profundidad
    # Estimado por proporción: ~50% de los 11m del área grande (reglamento fútbol 7/9)
    cv2.rectangle(img, to_px(14.0, 68.0), to_px(27.0, 62.5), c_white, lw, lineType=cv2.LINE_AA)
    
    # Medialuna superior (Arco de área)
    center_r = to_px(20.5, 59.0)
    cv2.circle(img, center_r, 2, c_white, -1, lineType=cv2.LINE_AA)  # Punto de penal superior
    # Ángulo centrado hacia abajo (90 grados en OpenCV es abajo)
    cv2.ellipse(img, center_r, (r_center, r_center), 0, 37, 143, c_white, lw, lineType=cv2.LINE_AA)

    # 6. Porterías (Protrusión exterior de 1.5 m)
    g_depth_px = int(1.5 * scale)
    
    # Portería inferior (Y=0, extendiendo a Y=-1.5)
    p_inf_1 = to_px(17.0, 0.0)
    p_inf_2 = to_px(24.0, -1.5)
    cv2.rectangle(img, p_inf_1, p_inf_2, c_white, lw, lineType=cv2.LINE_AA)
    
    # Portería superior (Y=68, extendiendo a Y=69.5)
    p_sup_1 = to_px(17.0, 68.0)
    p_sup_2 = to_px(24.0, 69.5)
    cv2.rectangle(img, p_sup_1, p_sup_2, c_white, lw, lineType=cv2.LINE_AA)

    # 7. Semicírculos de esquina (Córners de radio 1m)
    r_corner = int(1.0 * scale)
    cv2.ellipse(img, to_px(0.0, 0.0), (r_corner, r_corner), 0, 270, 360, c_white, lw, lineType=cv2.LINE_AA)
    cv2.ellipse(img, to_px(41.0, 0.0), (r_corner, r_corner), 0, 180, 270, c_white, lw, lineType=cv2.LINE_AA)
    cv2.ellipse(img, to_px(0.0, 68.0), (r_corner, r_corner), 0, 0, 90, c_white, lw, lineType=cv2.LINE_AA)
    cv2.ellipse(img, to_px(41.0, 68.0), (r_corner, r_corner), 0, 90, 180, c_white, lw, lineType=cv2.LINE_AA)

    return img, (w_px, h_px)


def draw_keypoints_status_2d(
    pitch_img: np.ndarray,
    estimator: HomographyEstimator,
    scale: float = 10.0,
    padding: int = 25
) -> np.ndarray:
    """
    Dibuja los puntos del modelo geométrico sobre la imagen de la cancha 2D vertical,
    coloreándolos según su estado de detección y uso en la homografía.
    
    - Verde: Detectado y utilizado como inlier en el cálculo.
    - Rojo: Detectado pero descartado (outlier).
    - Gris: No detectado o descartado por baja confianza.
    
    Args:
        pitch_img: Imagen base de la cancha generada por draw_pitch_2d.
        estimator: Estimador de homografía que contiene el historial de inliers.
        scale: Factor de escala de píxeles por metro.
        padding: Margen exterior en píxeles.
        
    Returns:
        np.ndarray: Imagen con los puntos clave superpuestos.
    """
    out_img = pitch_img.copy()
    
    # Colores BGR
    c_inlier = (0, 255, 0)      # Verde
    c_outlier = (0, 0, 255)     # Rojo
    c_undetected = (140, 140, 140)  # Gris

    for name, real_pos in FIELD_POINTS.items():
        # Calcular posición en píxeles usando la función auxiliar compartida
        px, py = meters_to_px(real_pos[0], real_pos[1], scale, padding)
        
        # Determinar estado
        was_detected = estimator.last_detected.get(name, False)
        was_inlier = estimator.last_inliers.get(name, False)
        
        if was_detected:
            if was_inlier:
                color = c_inlier
                label_color = (0, 100, 0)
                radius = 7
            else:
                color = c_outlier
                label_color = (0, 0, 150)
                radius = 6
        else:
            color = c_undetected
            label_color = (80, 80, 80)
            radius = 4
            
        # Dibujar punto clave en el canvas
        cv2.circle(out_img, (px, py), radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(out_img, (px, py), radius, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        
        # Escribir la etiqueta del punto (ej. "KP00") discretamente al lado
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale_font = 0.35
        cv2.putText(
            out_img, name, (px + 6, py + 4),
            font, scale_font, label_color, 1, lineType=cv2.LINE_AA
        )
        
    return out_img
