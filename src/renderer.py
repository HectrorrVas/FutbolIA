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

    def draw_field(self, w, h):
        """Dibuja o carga un campo de futbol cenital en orientacion VERTICAL."""
        from pathlib import Path
        
        # Intentar cargar la imagen personalizada del usuario en config/field.png o field.jpg
        field_img_paths = [
            Path("/content/Futbol2026/config/field.png"),
            Path("/content/Futbol2026/config/field.jpg"),
            Path("config/field.png"),
            Path("config/field.jpg"),
            Path(__file__).resolve().parent.parent / "config" / "field.png",
            Path(__file__).resolve().parent.parent / "config" / "field.jpg",
        ]
        
        for path in field_img_paths:
            if path.exists():
                img = cv2.imread(str(path))
                if img is not None:
                    img = cv2.resize(img, (w, h))
                    pad = int(w * 0.05)
                    fw, fh = w - 2 * pad, h - 2 * pad
                    return img, pad, fw, fh
                    
        # Fallback: dibujar campo dinamicamente si no existe imagen personalizada
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Color de cesped verde premium
        img[:] = (35, 145, 60)
        
        pad = 24
        fw, fh = w - 2 * pad, h - 2 * pad
        c = (255, 255, 255)
        lw = 2
        
        # 1. Borde del campo
        cv2.rectangle(img, (pad, pad), (pad + fw, pad + fh), c, lw, lineType=cv2.LINE_AA)
        
        # 2. Linea central (Horizontal, ya que el campo es vertical)
        mid_y = pad + fh // 2
        cv2.line(img, (pad, mid_y), (pad + fw, mid_y), c, lw, lineType=cv2.LINE_AA)
        
        # 3. Circulo central y punto central
        r_center = int(min(fw, fh) * 0.09)
        cv2.circle(img, (w // 2, mid_y), r_center, c, lw, lineType=cv2.LINE_AA)
        cv2.circle(img, (w // 2, mid_y), 3, c, -1, lineType=cv2.LINE_AA)
        
        # 4. Areas de penalti (superior e inferior)
        paw = int(fw * 0.68)      # ancho area grande
        pad_box = int(fh * 0.15)  # alto area grande
        px = pad + (fw - paw) // 2
        
        # Area superior grande
        cv2.rectangle(img, (px, pad), (px + paw, pad + pad_box), c, lw, lineType=cv2.LINE_AA)
        # Area inferior grande
        cv2.rectangle(img, (px, pad + fh - pad_box), (px + paw, pad + fh), c, lw, lineType=cv2.LINE_AA)
        
        # 5. Areas pequeñas
        gaw = int(fw * 0.32)      # ancho area chica
        gad = int(fh * 0.05)      # alto area chica
        gx = pad + (fw - gaw) // 2
        
        # Area superior chica
        cv2.rectangle(img, (gx, pad), (gx + gaw, pad + gad), c, lw, lineType=cv2.LINE_AA)
        # Area inferior chica
        cv2.rectangle(img, (gx, pad + fh - gad), (gx + gaw, pad + fh), c, lw, lineType=cv2.LINE_AA)
        
        # 6. Puntos de penalti y arcos de area
        r_arc = int(min(fw, fh) * 0.08)
        pen_spot_dist = int(fh * 0.10)
        
        # Punto de penalti superior
        psy = pad + pen_spot_dist
        cv2.circle(img, (w // 2, psy), 3, c, -1, lineType=cv2.LINE_AA)
        # Arco superior
        cv2.ellipse(img, (w // 2, psy), (r_arc, r_arc), 0, 35, 145, c, lw, lineType=cv2.LINE_AA)
        
        # Punto de penalti inferior
        piy = pad + fh - pen_spot_dist
        cv2.circle(img, (w // 2, piy), 3, c, -1, lineType=cv2.LINE_AA)
        # Arco inferior
        cv2.ellipse(img, (w // 2, piy), (r_arc, r_arc), 0, 215, 325, c, lw, lineType=cv2.LINE_AA)
        
        # 7. Porterias (Nets que sobresalen del campo)
        gw = int(fw * 0.18)  # ancho de la porteria
        gd = int(pad * 0.8)  # profundidad de la porteria
        glx = w // 2 - gw // 2
        grx = w // 2 + gw // 2
        
        # Porteria superior
        cv2.rectangle(img, (glx, pad - gd), (grx, pad), c, lw, lineType=cv2.LINE_AA)
        # Red (lineas grises internas)
        for rx in range(glx + 4, grx, 6):
            cv2.line(img, (rx, pad - gd), (rx, pad), (150, 180, 150), 1, lineType=cv2.LINE_AA)
        for ry in range(pad - gd + 4, pad, 4):
            cv2.line(img, (glx, ry), (grx, ry), (150, 180, 150), 1, lineType=cv2.LINE_AA)
            
        # Porteria inferior
        cv2.rectangle(img, (glx, pad + fh), (grx, pad + fh + gd), c, lw, lineType=cv2.LINE_AA)
        # Red
        for rx in range(glx + 4, grx, 6):
            cv2.line(img, (rx, pad + fh), (rx, pad + fh + gd), (150, 180, 150), 1, lineType=cv2.LINE_AA)
        for ry in range(pad + fh + 4, pad + fh + gd, 4):
            cv2.line(img, (glx, ry), (grx, ry), (150, 180, 150), 1, lineType=cv2.LINE_AA)

        # 8. Semicirculos de esquina (corners)
        rc = int(min(fw, fh) * 0.02)
        cv2.ellipse(img, (pad, pad), (rc, rc), 0, 0, 90, c, lw, lineType=cv2.LINE_AA)
        cv2.ellipse(img, (pad + fw, pad), (rc, rc), 0, 90, 180, c, lw, lineType=cv2.LINE_AA)
        cv2.ellipse(img, (pad, pad + fh), (rc, rc), 0, 270, 360, c, lw, lineType=cv2.LINE_AA)
        cv2.ellipse(img, (pad + fw, pad + fh), (rc, rc), 0, 180, 270, c, lw, lineType=cv2.LINE_AA)
        
        return img, pad, fw, fh

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

    def render_canvas(self, frame, players, ball_pos, draw_coordination=True):
        """
        Dibuja los marcadores en el frame y genera el mapa tactico 2D vertical side-by-side.
        Retorna la imagen final del canvas compuesto.
        """
        vid_h, vid_w = frame.shape[:2]
        
        teamA_video_points = []
        teamB_video_points = []

        # 1. Dibujar marcadores sobre el frame de video original
        video_out = frame.copy()
        for p in players:
            cx, cy = p["center"]
            color = self.class_colors.get(p["class_id"], (150, 150, 150))
            
            # Dibujar el marcador
            cv2.circle(video_out, (int(cx), int(cy)), self.marker_radius, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(video_out, (int(cx), int(cy)), self.marker_radius, (0, 0, 0), int(self.outline_thickness), lineType=cv2.LINE_AA)

            # Recolectar puntos reales en video para la malla tactica
            if p["class_id"] == 2:
                teamA_video_points.append((int(cx), int(cy)))
            elif p["class_id"] == 3:
                teamB_video_points.append((int(cx), int(cy)))
            
        if ball_pos is not None:
            bx, by = ball_pos
            cv2.circle(video_out, (int(bx), int(by)), self.marker_radius - 1, self.class_colors[1], -1, lineType=cv2.LINE_AA)
            cv2.circle(video_out, (int(bx), int(by)), self.marker_radius - 1, (0, 0, 0), int(self.outline_thickness), lineType=cv2.LINE_AA)

        # Dibujar lineas de coordinacion y bloques defensivos DIRECTAMENTE en la representacion real (Video)
        if draw_coordination:
            from src.analytics import TacticalAnalyzer
            # Bloques defensivos (Convex Hull) sobre el video
            TacticalAnalyzer.draw_defensive_block(video_out, teamA_video_points, self.class_colors[2])
            TacticalAnalyzer.draw_defensive_block(video_out, teamB_video_points, self.class_colors[3])
            # Lineas de coordinacion (Mallas de vecindad) sobre el video
            TacticalAnalyzer.draw_coordination_mesh(video_out, teamA_video_points, self.class_colors[2], k_neighbors=2)
            TacticalAnalyzer.draw_coordination_mesh(video_out, teamB_video_points, self.class_colors[3], k_neighbors=2)

        # 2. Renderizar el Mapa Tactico 2D (Limpio de lineas, solo marcadores e IDs)
        map_w = int(vid_w * 0.38)
        tac, f_pad, f_fw, f_fh = self.draw_field(map_w, vid_h)
        
        mapped_players = []

        # Mapear posiciones de los jugadores en el mapa 2D
        for p in players:
            cx, cy = p["center"]
            # Escalar linealmente de las coordenadas del video al mapa
            mx = f_pad + int((cx / vid_w) * f_fw)
            my = f_pad + int((cy / vid_h) * f_fh)
            color = self.class_colors.get(p["class_id"], (150, 150, 150))
                
            # Dibujar circulo en mapa 2D
            cv2.circle(tac, (mx, my), 8, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(tac, (mx, my), 8, (0, 0, 0), 1, lineType=cv2.LINE_AA)

            track_id = p.get("track_id")
            mapped_players.append({
                "track_id": track_id,
                "class_id": p["class_id"],
                "center_map": (mx, my)
            })
            
        if ball_pos is not None:
            bx, by = ball_pos
            mx = f_pad + int((bx / vid_w) * f_fw)
            my = f_pad + int((by / vid_h) * f_fh)
            cv2.circle(tac, (mx, my), 7, self.class_colors[1], -1, lineType=cv2.LINE_AA)
            cv2.circle(tac, (mx, my), 7, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        # 3. Ensamblar Panel izquierdo (video) + Panel derecho (mapa) + Barra inferior (leyenda)
        top_row = np.hstack([video_out, tac])
        legend_bar = self.draw_legend(top_row.shape[1])
        canvas = np.vstack([top_row, legend_bar])
        
        return canvas, mapped_players
