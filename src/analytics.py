import cv2
import numpy as np
from pathlib import Path


class HeatmapManager:
    """Clase encargada de registrar posiciones de jugadores y generar mapas de calor."""
    def __init__(self, output_dir="output/heatmaps"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Diccionario para almacenar coordenadas por track_id: {track_id: [(x, y), ...]}
        self.positions = {}
        # Diccionario para almacenar coordenadas por equipo (class_id): {class_id: [(x, y), ...]}
        self.team_positions = {2: [], 3: []}  # 2: EquipoA, 3: EquipoB

    def update(self, players_map_coords):
        """Registra las coordenadas mapeadas de los jugadores en este frame."""
        for p in players_map_coords:
            track_id = p.get("track_id")
            class_id = p["class_id"]
            mx, my = p["center_map"]

            # Guardar por jugador individual si tiene track ID
            if track_id is not None:
                if track_id not in self.positions:
                    self.positions[track_id] = []
                self.positions[track_id].append((mx, my))

            # Guardar por equipo
            if class_id in self.team_positions:
                self.team_positions[class_id].append((mx, my))

    def generate_and_save_heatmaps(self, renderer, map_w, map_h, specific_player_ids=None):
        """Genera y guarda los mapas de calor en disco."""
        # Generar imagen de cancha base limpia
        field_img = renderer.draw_field(map_w, map_h)
        if isinstance(field_img, tuple):
            field_img = field_img[0]

        print(f"\nGenerando mapas de calor en '{self.output_dir}'...")

        # 1. Mapas de calor individuales
        player_ids_to_process = specific_player_ids or list(self.positions.keys())
        for pid in player_ids_to_process:
            coords = self.positions.get(pid, [])
            if len(coords) < 10:  # Evitar mapas vacios o muy cortos
                continue
            output_file = self.output_dir / f"jugador_id_{pid}.png"
            self._save_heatmap(field_img, coords, output_file)

        # 2. Mapas de calor por equipos
        team_names = {2: "Equipo_A", 3: "Equipo_B"}
        for cid, name in team_names.items():
            coords = self.team_positions.get(cid, [])
            if len(coords) < 15:
                continue
            output_file = self.output_dir / f"mapa_calor_{name}.png"
            self._save_heatmap(field_img, coords, output_file)

        print(f"[OK] Mapas de calor guardados en: {self.output_dir}")

    def _save_heatmap(self, field_image, coords, output_path):
        h, w = field_image.shape[:2]
        accum = np.zeros((h, w), dtype=np.float32)

        for x, y in coords:
            ix, iy = int(x), int(y)
            if 0 <= ix < w and 0 <= iy < h:
                accum[iy, ix] += 1.0

        blur_radius = int(w * 0.08)
        if blur_radius % 2 == 0:
            blur_radius += 1
        density = cv2.GaussianBlur(accum, (blur_radius, blur_radius), 0)

        max_val = np.max(density)
        if max_val > 0:
            density = (density / max_val * 255).astype(np.uint8)
        else:
            return

        heatmap_color = cv2.applyColorMap(density, cv2.COLORMAP_JET)
        mask = (density > 12).astype(np.uint8)

        result = field_image.copy()
        for c in range(3):
            result[:, :, c] = np.where(
                mask == 1,
                cv2.addWeighted(heatmap_color[:, :, c], 0.65, field_image[:, :, c], 0.35, 0),
                field_image[:, :, c]
            )

        cv2.imwrite(str(output_path), result)


class TacticalAnalyzer:
    """
    Clase encargada del analisis de lineas de coordinacion y bloques defensivos.
    
    Escala de metros estimada:
      - Se asume que el campo visible ocupa ~100m de largo y ~65m de ancho.
      - La escala se calcula dinamicamente por frame segun el tamano del video.
    """

    # Dimensiones reales aproximadas del campo en metros
    FIELD_LENGTH_M = 100.0
    FIELD_WIDTH_M  = 65.0

    @staticmethod
    def _px_to_meters(dist_px, vid_w, vid_h):
        """Convierte distancia en pixeles a metros usando escala estimada del campo."""
        scale_x = TacticalAnalyzer.FIELD_LENGTH_M / vid_w
        scale_y = TacticalAnalyzer.FIELD_WIDTH_M  / vid_h
        scale   = (scale_x + scale_y) / 2.0
        return dist_px * scale

    @staticmethod
    def _draw_distance_label(img, p1, p2, dist_m):
        """Dibuja la distancia en metros sobre la linea que conecta dos jugadores."""
        mid_x = int((p1[0] + p2[0]) / 2)
        mid_y = int((p1[1] + p2[1]) / 2)
        label = f"{dist_m:.1f}m"
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.38
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)

        # Fondo negro semitransparente
        pad = 3
        overlay = img.copy()
        cv2.rectangle(
            overlay,
            (mid_x - tw // 2 - pad, mid_y - th - pad),
            (mid_x + tw // 2 + pad, mid_y + pad),
            (0, 0, 0), -1
        )
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

        # Texto blanco
        cv2.putText(
            img, label,
            (mid_x - tw // 2, mid_y),
            font, scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA
        )

    @staticmethod
    def draw_coordination_mesh(img, points, color, k_neighbors=2,
                               show_distances=False, vid_w=1280, vid_h=720):
        """
        Dibuja lineas de conexion entre los jugadores mas cercanos del mismo equipo.
        
        Args:
            img           : Frame donde dibujar.
            points        : Lista de (x, y) en coordenadas de video.
            color         : Color BGR del equipo.
            k_neighbors   : Cuantos vecinos conectar por jugador.
            show_distances: Si True, dibuja la distancia en metros sobre cada linea.
            vid_w, vid_h  : Dimensiones del video para calcular escala.
        """
        if len(points) < 2:
            return

        pts  = np.array(points, dtype=np.int32)
        seen = set()  # Evitar dibujar el mismo par dos veces

        for i, p1 in enumerate(pts):
            dists = []
            for j, p2 in enumerate(pts):
                if i == j:
                    continue
                dist_px = float(np.linalg.norm(p1.astype(float) - p2.astype(float)))
                dists.append((dist_px, j))

            dists.sort()
            for dist_px, j in dists[:k_neighbors]:
                key = (min(i, j), max(i, j))
                if key in seen:
                    continue
                seen.add(key)

                p2 = pts[j]

                # Linea punteada/solida premium
                cv2.line(
                    img,
                    (int(p1[0]), int(p1[1])),
                    (int(p2[0]), int(p2[1])),
                    color, 1, lineType=cv2.LINE_AA
                )

                if show_distances:
                    dist_m = TacticalAnalyzer._px_to_meters(dist_px, vid_w, vid_h)
                    TacticalAnalyzer._draw_distance_label(img, p1, p2, dist_m)

    @staticmethod
    def draw_defensive_block(img, points, color, alpha=0.14):
        """Calcula y dibuja el bloque defensivo (Convex Hull) semitransparente del equipo."""
        if len(points) < 3:
            return

        pts  = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(pts)

        # Relleno semitransparente
        overlay = img.copy()
        cv2.drawContours(overlay, [hull], -1, color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Borde solido del bloque
        cv2.drawContours(img, [hull], -1, color, 1, lineType=cv2.LINE_AA)

    @staticmethod
    def render_team_focus(frame, all_players, ball_pos, focused_class_id,
                          class_colors, marker_radius=7, show_distances=True):
        """
        Genera un frame de analisis centrado en UN equipo.

        - El equipo enfocado: marcadores normales + malla de coordinacion + bloque + distancias.
        - Los demas jugadores: atenuados al 30% y sin lineas, para mantener el contexto.
        - Balon: siempre visible.

        Args:
            frame            : Frame BGR original del video.
            all_players      : Lista de dicts con keys: class_id, center, track_id.
            ball_pos         : (cx, cy) del balon o None.
            focused_class_id : 2 para EquipoA, 3 para EquipoB.
            class_colors     : Diccionario {class_id: (B, G, R)}.
            marker_radius    : Radio del marcador del jugador enfocado.
            show_distances   : Si True, dibuja distancias en metros en las lineas.

        Returns:
            Frame BGR con el overlay de analisis del equipo enfocado.
        """
        vid_h, vid_w = frame.shape[:2]
        out = frame.copy()

        focused_points  = []
        focused_players = []

        # 1. Atenuar todos los jugadores que NO son del equipo enfocado
        for p in all_players:
            cx, cy = p["center"]
            cid    = p["class_id"]
            color  = class_colors.get(cid, (150, 150, 150))

            if cid != focused_class_id:
                # Dibujar marcador atenuado (50% opacidad = mezcla con el fondo)
                overlay = out.copy()
                cv2.circle(overlay, (int(cx), int(cy)), marker_radius - 1, color, -1, lineType=cv2.LINE_AA)
                cv2.circle(overlay, (int(cx), int(cy)), marker_radius - 1, (0, 0, 0), 1,  lineType=cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
            else:
                focused_points.append((int(cx), int(cy)))
                focused_players.append(p)

        # 2. Dibujar bloque defensivo / ofensivo del equipo enfocado
        team_color = class_colors.get(focused_class_id, (255, 255, 255))
        TacticalAnalyzer.draw_defensive_block(out, focused_points, team_color, alpha=0.16)

        # 3. Dibujar malla de coordinacion con distancias
        TacticalAnalyzer.draw_coordination_mesh(
            out, focused_points, team_color,
            k_neighbors=2,
            show_distances=show_distances,
            vid_w=vid_w, vid_h=vid_h
        )

        # 4. Dibujar marcadores premium del equipo enfocado (encima de las lineas)
        for p in focused_players:
            cx, cy   = p["center"]

            # Sombra del circulo
            cv2.circle(out, (int(cx) + 2, int(cy) + 2), marker_radius + 1,
                       (0, 0, 0), -1, lineType=cv2.LINE_AA)
            # Circulo relleno color equipo
            cv2.circle(out, (int(cx), int(cy)), marker_radius,
                       team_color, -1, lineType=cv2.LINE_AA)
            # Borde blanco interior
            cv2.circle(out, (int(cx), int(cy)), marker_radius,
                       (255, 255, 255), 1, lineType=cv2.LINE_AA)

        # 5. Dibujar el balon (siempre visible)
        if ball_pos is not None:
            bx, by = ball_pos
            ball_color = class_colors.get(1, (255, 255, 255))
            cv2.circle(out, (int(bx) + 1, int(by) + 1), 6, (0, 0, 0), -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (int(bx), int(by)), 5, ball_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (int(bx), int(by)), 5, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        # 6. Watermark del equipo analizado (esquina superior izquierda)
        team_label = "EQUIPO A - ANALISIS TACTICO" if focused_class_id == 2 else "EQUIPO B - ANALISIS TACTICO"
        font   = cv2.FONT_HERSHEY_SIMPLEX
        tscale = 0.6
        tthick = 2
        (lw, lh), _ = cv2.getTextSize(team_label, font, tscale, tthick)
        # Fondo del label
        cv2.rectangle(out, (14, 14), (lw + 26, lh + 26), (0, 0, 0), -1)
        cv2.rectangle(out, (14, 14), (lw + 26, lh + 26), team_color, 1)
        cv2.putText(out, team_label, (20, lh + 19), font, tscale, team_color, tthick, cv2.LINE_AA)

        return out
