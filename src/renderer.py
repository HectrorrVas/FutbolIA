import cv2
import numpy as np

class Renderer:
    """Clase encargada de dibujar los overlays y marcadores en los frames."""
    def __init__(self, class_colors=None):
        # Mapeo por defecto de clase a color BGR
        self.class_colors = class_colors or {
            0: (0, 215, 255),    # Arbitro -> Amarillo
            1: (255, 255, 255),  # Balon -> Blanco
            2: (220, 60, 30),    # EquipoA -> Azul (BGR: B=220, G=60, R=30)
            3: (30, 30, 220),    # EquipoB -> Rojo (BGR: B=30, G=30, R=220)
            5: (30, 180, 30),    # Portero -> Verde
        }
        self.marker_radius = 6
        self.outline_thickness = 1.5
        self.legend_h = 60

    def real_to_canvas_px(self, X_m: float, Y_m: float, map_w: int, map_h: int) -> tuple:
        """
        Convierte coordenadas reales en metros a píxeles en el canvas del mapa táctico.
        Mapea el campo de 41m x 68m centrándolo con padding.
        """
        padding = int(map_w * 0.06)
        scale_x = (map_w - 2 * padding) / 41.0
        scale_y = (map_h - 2 * padding) / 68.0
        scale = min(scale_x, scale_y)
        
        pad_x = (map_w - 41.0 * scale) / 2.0
        pad_y = (map_h - 68.0 * scale) / 2.0
        
        px = int(pad_x + X_m * scale)
        py = int(pad_y + (68.0 - Y_m) * scale)  # Inversión de Y
        return px, py

    def draw_field(self, w, h):
        """Dibuja el campo de fútbol de manera dinámica y métricamente exacta."""
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Cesped verde premium
        img[:] = (35, 145, 60)
        
        c = (255, 255, 255)
        lw = 2
        
        def to_px(x, y):
            return self.real_to_canvas_px(x, y, w, h)
            
        # 1. Borde del campo (0,0) a (41,68)
        cv2.rectangle(img, to_px(0.0, 0.0), to_px(41.0, 68.0), c, lw, lineType=cv2.LINE_AA)
        
        # 2. Línea central
        cv2.line(img, to_px(0.0, 34.0), to_px(41.0, 34.0), c, lw, lineType=cv2.LINE_AA)
        
        # 3. Círculo central y punto central
        # El radio es de 5.0 metros reales
        p_center = to_px(20.5, 34.0)
        p_edge = to_px(20.5, 39.0)
        r_center = int(abs(p_center[1] - p_edge[1]))
        
        cv2.circle(img, p_center, r_center, c, lw, lineType=cv2.LINE_AA)
        cv2.circle(img, p_center, 3, c, -1, lineType=cv2.LINE_AA)
        
        # 4. Área penal grande inferior (7,0) a (34,11)
        cv2.rectangle(img, to_px(7.0, 0.0), to_px(34.0, 11.0), c, lw, lineType=cv2.LINE_AA)
        
        # Área chica inferior (14,0) a (27,5.5)
        cv2.rectangle(img, to_px(14.0, 0.0), to_px(27.0, 5.5), c, lw, lineType=cv2.LINE_AA)
        
        # Medialuna inferior (arco de área) centrado en punto penal (20.5, 9.0)
        center_l = to_px(20.5, 9.0)
        cv2.circle(img, center_l, 2, c, -1, lineType=cv2.LINE_AA)
        cv2.ellipse(img, center_l, (r_center, r_center), 0, 217, 323, c, lw, lineType=cv2.LINE_AA)
        
        # 5. Área penal grande superior (7,68) a (34,57)
        cv2.rectangle(img, to_px(7.0, 68.0), to_px(34.0, 57.0), c, lw, lineType=cv2.LINE_AA)
        
        # Área chica superior (14,68) a (27,62.5)
        cv2.rectangle(img, to_px(14.0, 68.0), to_px(27.0, 62.5), c, lw, lineType=cv2.LINE_AA)
        
        # Medialuna superior centrado en punto penal (20.5, 59.0)
        center_r = to_px(20.5, 59.0)
        cv2.circle(img, center_r, 2, c, -1, lineType=cv2.LINE_AA)
        cv2.ellipse(img, center_r, (r_center, r_center), 0, 37, 143, c, lw, lineType=cv2.LINE_AA)
        
        # 6. Porterías (sobresalen exteriormente 1.5m)
        cv2.rectangle(img, to_px(17.0, 0.0), to_px(24.0, -1.5), c, lw, lineType=cv2.LINE_AA)
        cv2.rectangle(img, to_px(17.0, 68.0), to_px(24.0, 69.5), c, lw, lineType=cv2.LINE_AA)
        
        # Red para porterías (decorativo rápido)
        # Superior
        p1_sup = to_px(17.0, 68.0)
        p2_sup = to_px(24.0, 69.5)
        for rx in range(p1_sup[0] + 4, p2_sup[0], 6):
            cv2.line(img, (rx, p1_sup[1]), (rx, p2_sup[1]), (150, 180, 150), 1, lineType=cv2.LINE_AA)
        for ry in range(p1_sup[1] + 4, p2_sup[1], 4):
            cv2.line(img, (p1_sup[0], ry), (p2_sup[0], ry), (150, 180, 150), 1, lineType=cv2.LINE_AA)
        # Inferior
        p1_inf = to_px(17.0, 0.0)
        p2_inf = to_px(24.0, -1.5)
        for rx in range(p1_inf[0] + 4, p2_inf[0], 6):
            cv2.line(img, (rx, p1_inf[1]), (rx, p2_inf[1]), (150, 180, 150), 1, lineType=cv2.LINE_AA)
        for ry in range(p2_inf[1] + 4, p1_inf[1], 4):
            cv2.line(img, (p1_inf[0], ry), (p2_inf[0], ry), (150, 180, 150), 1, lineType=cv2.LINE_AA)

        # 7. Corners (radio 1m)
        p_c1 = to_px(0.0, 0.0)
        p_c2 = to_px(41.0, 0.0)
        p_c3 = to_px(0.0, 68.0)
        p_c4 = to_px(41.0, 68.0)
        r_corner = int(abs(to_px(0.0, 0.0)[0] - to_px(1.0, 0.0)[0]))
        
        cv2.ellipse(img, p_c1, (r_corner, r_corner), 0, 270, 360, c, lw, lineType=cv2.LINE_AA)
        cv2.ellipse(img, p_c2, (r_corner, r_corner), 0, 180, 270, c, lw, lineType=cv2.LINE_AA)
        cv2.ellipse(img, p_c3, (r_corner, r_corner), 0, 0, 90, c, lw, lineType=cv2.LINE_AA)
        cv2.ellipse(img, p_c4, (r_corner, r_corner), 0, 90, 180, c, lw, lineType=cv2.LINE_AA)
        
        padding = int(w * 0.06)
        return img, padding, w - 2 * padding, h - 2 * padding

    def draw_legend(self, canvas_w):
        """Dibuja la barra de leyenda inferior."""
        bar = np.zeros((self.legend_h, canvas_w, 3), dtype=np.uint8)
        bar[:] = (20, 20, 20)
        items = [
            ("Equipo A", self.class_colors[2]),
            ("Equipo B", self.class_colors[3]),
            ("Arbitro",  self.class_colors[0]),
            ("Portero",  self.class_colors[5]),
            ("Balon",    self.class_colors[1]),
        ]
        spacing = canvas_w // len(items)
        y = self.legend_h // 2
        for i, (name, color) in enumerate(items):
            x = spacing * i + spacing // 2
            cv2.circle(bar, (x - 55, y), 12, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(bar, (x - 55, y), 12, (40, 40, 40), 1, lineType=cv2.LINE_AA)
            cv2.putText(bar, name, (x - 35, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1, lineType=cv2.LINE_AA)
        return bar

    def draw_id_label(self, img, cx, cy, track_id):
        """Dibuja el ID del jugador centrado arriba del marcador circular."""
        text = str(track_id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.38
        thickness = 1
        (t_w, t_h), _ = cv2.getTextSize(text, font, scale, thickness)
        tx = int(cx - t_w / 2)
        ty = int(cy - self.marker_radius - 4)
        
        # Dibujar borde negro para contraste
        cv2.putText(img, text, (tx, ty), font, scale, (0, 0, 0), thickness + 2, lineType=cv2.LINE_AA)
        # Dibujar texto blanco
        cv2.putText(img, text, (tx, ty), font, scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

    def render_canvas(self, frame, players, ball_pos, estimator=None, draw_coordination=True):
        """
        Dibuja los marcadores en el frame y genera el mapa táctico 2D vertical de forma métrica
        mediante homografía. Retorna la imagen final del canvas compuesto.
        """
        vid_h, vid_w = frame.shape[:2]
        
        teamA_video_points = []
        teamB_video_points = []

        # 1. Dibujar marcadores e IDs sobre el frame de video original
        video_out = frame.copy()
        for p in players:
            cx, cy = p["center"]
            color = self.class_colors.get(p["class_id"], (150, 150, 150))
            
            # Dibujar el marcador
            cv2.circle(video_out, (int(cx), int(cy)), self.marker_radius, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(video_out, (int(cx), int(cy)), self.marker_radius, (0, 0, 0), int(self.outline_thickness), lineType=cv2.LINE_AA)

            # Dibujar ID persistente del jugador
            track_id = p.get("track_id")
            if track_id is not None:
                self.draw_id_label(video_out, cx, cy, track_id)

            # Recolectar puntos de video para la malla táctica
            if p["class_id"] == 2:
                teamA_video_points.append((int(cx), int(cy)))
            elif p["class_id"] == 3:
                teamB_video_points.append((int(cx), int(cy)))
            
        if ball_pos is not None:
            bx, by = ball_pos
            cv2.circle(video_out, (int(bx), int(by)), self.marker_radius - 1, self.class_colors[1], -1, lineType=cv2.LINE_AA)
            cv2.circle(video_out, (int(bx), int(by)), self.marker_radius - 1, (0, 0, 0), int(self.outline_thickness), lineType=cv2.LINE_AA)

        # Dibujar líneas de coordinación y bloques defensivos sobre el video
        if draw_coordination:
            from src.analytics import TacticalAnalyzer
            TacticalAnalyzer.draw_defensive_block(video_out, teamA_video_points, self.class_colors[2])
            TacticalAnalyzer.draw_defensive_block(video_out, teamB_video_points, self.class_colors[3])
            TacticalAnalyzer.draw_coordination_mesh(video_out, teamA_video_points, self.class_colors[2], k_neighbors=2)
            TacticalAnalyzer.draw_coordination_mesh(video_out, teamB_video_points, self.class_colors[3], k_neighbors=2)

        # 2. Renderizar el Mapa Táctico 2D (Usando coordenadas reales en metros)
        map_w = int(vid_w * 0.38)
        tac, f_pad, f_fw, f_fh = self.draw_field(map_w, vid_h)
        
        mapped_players = []

        # Mapear posiciones reales de los jugadores
        for p in players:
            real_pos = p.get("real_pos")
            if real_pos is not None:
                X_m, Y_m = real_pos
                mx, my = self.real_to_canvas_px(X_m, Y_m, map_w, vid_h)
            else:
                cx, cy = p["center"]
                mx = f_pad + int((cx / vid_w) * f_fw)
                my = f_pad + int((cy / vid_h) * f_fh)
                
            color = self.class_colors.get(p["class_id"], (150, 150, 150))
            cv2.circle(tac, (mx, my), 8, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(tac, (mx, my), 8, (0, 0, 0), 1, lineType=cv2.LINE_AA)

            track_id = p.get("track_id")
            if track_id is not None:
                self.draw_id_label(tac, mx, my, track_id)

            mapped_players.append({
                "track_id": track_id,
                "class_id": p["class_id"],
                "center_map": (mx, my)
            })
            
        # Mapear la pelota
        if ball_pos is not None:
            real_ball = None
            if estimator is not None and estimator.H is not None:
                try:
                    real_ball = estimator.transform_image_to_real([ball_pos])[0]
                except:
                    pass
            
            if real_ball is not None:
                bx, by = self.real_to_canvas_px(real_ball[0], real_ball[1], map_w, vid_h)
            else:
                bx, by = ball_pos
                bx = f_pad + int((bx / vid_w) * f_fw)
                by = f_pad + int((by / vid_h) * f_fh)
                
            cv2.circle(tac, (bx, by), 7, self.class_colors[1], -1, lineType=cv2.LINE_AA)
            cv2.circle(tac, (bx, by), 7, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        # 3. Ensamblar Panel izquierdo (video) + Panel derecho (mapa) + Barra inferior (leyenda)
        top_row = np.hstack([video_out, tac])
        legend_bar = self.draw_legend(top_row.shape[1])
        canvas = np.vstack([top_row, legend_bar])
        
        return canvas, mapped_players

