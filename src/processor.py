import cv2
import numpy as np
import subprocess
from pathlib import Path
from tqdm import tqdm
import imageio_ffmpeg as im_ffmpeg

from ultralytics import YOLO
from src.tracking.tracker import FootballTracker
from src.field_model import HomographyEstimator
from src.filters import BallKalmanFilter
from src.renderer import Renderer
from src.analytics import HeatmapManager, TacticalAnalyzer


def _encode_h264(input_path: Path, output_path: Path):
    """Re-codifica un archivo de video con FFmpeg a H.264 compatible."""
    ffmpeg_exe = im_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    input_path.unlink()


class VideoProcessor:
    """
    Clase principal que orquesta el pipeline de procesamiento de video.
    Utiliza YOLO Pose para la detección de la cancha, YOLO Detect para los jugadores y balón,
    y el FootballTracker personalizado para un ReID robusto y persistente.
    """

    def __init__(self, model_path, pose_model_path=None, confidence=0.25, imgsz=640):
        self.player_conf = confidence
        self.ball_conf   = 0.12   # Umbral bajo para el balón (pequeño y rápido)
        self.imgsz       = imgsz

        if pose_model_path is None:
            pose_model_path = str(Path(model_path).parent / "best_pose.pt")

        print(f"Cargando modelos YOLO...")
        self.yolo_detect = YOLO(model_path)
        self.yolo_pose   = YOLO(pose_model_path)
        print(f"  -> Detección: {Path(model_path).name}")
        print(f"  -> Pose Cancha: {Path(pose_model_path).name}")

        self.tracker = FootballTracker(min_player_conf=self.player_conf)
        self.estimator = HomographyEstimator(min_confidence=0.3)
        self.renderer = Renderer()
        self.heatmap_manager = HeatmapManager()
        self.ball_filter = None

    # ------------------------------------------------------------------ #
    # Escaneo rápido de IDs — Pase 1 (sin renderizado)                    #
    # ------------------------------------------------------------------ #
    def scan_player_ids(self, input_video_path: str, max_frames: int = None) -> dict:
        """
        Hace un pase rápido por el video sin renderizar ningún frame.
        Retorna un diccionario con los IDs de track encontrados y cuántos
        frames apareció cada uno: {track_id: {frames, class_id}}.
        """
        input_path = Path(input_video_path)
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"No se pudo abrir el video: {input_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames is None:
            max_frames = total_frames
        dt = 1.0 / fps if fps > 0 else 0.04

        print(f"\n🔍 Escaneando IDs de jugadores en '{input_path.name}'...")
        print(f"   Frames a procesar: {min(max_frames, total_frames)} / {total_frames}")

        id_registry = {}  # {track_id: {"frames": int, "class_id": int}}
        frame_idx = 0

        pbar = tqdm(total=min(max_frames, total_frames), unit="frame", ncols=80,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        try:
            while cap.isOpened() and frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                pbar.update(1)
                timestamp = frame_idx * dt

                # Detección de la cancha
                pose_results = self.yolo_pose(frame, verbose=False)
                detected_kps = {}
                if len(pose_results) > 0 and pose_results[0].keypoints is not None:
                    kpts = pose_results[0].keypoints
                    xy   = kpts.xy.cpu().numpy()[0]
                    conf = kpts.conf.cpu().numpy()[0] if kpts.conf is not None else [1.0] * len(xy)
                    for i in range(min(20, len(xy))):
                        detected_kps[f"KP{i:02d}"] = (float(xy[i][0]), float(xy[i][1]), float(conf[i]))
                self.estimator.estimate(detected_kps)

                # Detección de jugadores
                detect_results = self.yolo_detect(
                    frame, conf=self.player_conf, imgsz=self.imgsz, verbose=False
                )[0]

                # Actualizar tracker
                active_tracks = self.tracker.update(frame, detect_results, self.estimator, timestamp)

                # Registrar IDs activos
                for t in active_tracks:
                    tid = t["track_id"]
                    if tid not in id_registry:
                        id_registry[tid] = {"frames": 0, "class_id": t["class_id"]}
                    id_registry[tid]["frames"] += 1
        finally:
            cap.release()
            pbar.close()

        # Ordenar por cantidad de frames (los jugadores más visibles primero)
        id_registry = dict(sorted(id_registry.items(), key=lambda x: -x[1]["frames"]))
        return id_registry

    def process(self, input_video_path, output_video_path, mode="reid", player_id: int = None):
        """
        Procesa el video de entrada y genera los videos/imágenes según el modo:
          - 'reid'           : Video principal con mapa 2D y IDs persistentes.
          - 'teams'          : Videos de análisis por equipo (Equipo A y Equipo B).
          - 'full'           : Los 4 videos: reid + teams + grid 2x2.
          - 'player_heatmap' : Mapa de calor de un jugador específico (requiere player_id).
        """
        input_path  = Path(input_video_path)
        stem = input_path.stem
        if output_video_path is not None:
            output_path = Path(output_video_path)
            out_dir     = output_path.parent
        else:
            out_dir     = input_path.parent.parent / "processed"
            output_path = out_dir / f"{stem}_main.mp4"

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"No se pudo abrir el video: {input_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS)
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dt           = 1.0 / fps if fps > 0 else 0.04

        map_w    = int(width * 0.38)
        canvas_w = width + map_w
        canvas_h = height + self.renderer.legend_h

        DO_REID  = mode in ("reid", "full")
        DO_TEAMS = mode in ("teams", "full")
        DO_GRID  = mode == "full"
        DO_HEATMAP_PLAYER = mode == "player_heatmap"

        modo_label = {
            "reid":           "ReID + Mapa 2D",
            "teams":          "Análisis por Equipos",
            "full":           "Completo (4 videos)",
            "player_heatmap": f"Mapa de Calor — Jugador ID {player_id}"
        }.get(mode, mode)

        # ── Modo Mapa de Calor Individual: ruta dedicada ─────────────────────────
        if DO_HEATMAP_PLAYER:
            if player_id is None:
                raise ValueError("Debes especificar 'player_id' para el modo 'player_heatmap'.")
            self._run_player_heatmap(input_path, out_dir, stem, fps, total_frames, dt,
                                     map_w, height, player_id)
            return

        print("=" * 72)
        print("SISTEMA DE ANALISIS TACTICO DE FUTBOL - Futbol2026")
        print("=" * 72)
        print(f"  Entrada          : {input_path.name}")
        print(f"  Resolución       : {width}x{height}  |  FPS: {fps:.2f}  |  Frames: {total_frames}")
        print(f"  Modo             : {modo_label}")
        print(f"  Canvas combinado : {canvas_w}x{canvas_h}")
        print(f"  Salida           : {out_dir}")
        print()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Cuadrícula 2x2 (solo si mode=full)
        grid_w = width * 2
        grid_h = height * 2

        # Rutas de archivos temporales (raw antes de FFmpeg)
        tmp_main = out_dir / f"{stem}_main_tmp.mp4"
        tmp_a    = out_dir / f"{stem}_equipoA_tmp.mp4"
        tmp_b    = out_dir / f"{stem}_equipoB_tmp.mp4"
        tmp_grid = out_dir / f"{stem}_grid_tmp.mp4"

        # Rutas de archivos finales H.264
        out_main = out_dir / f"{stem}_main.mp4"
        out_a    = out_dir / f"{stem}_equipoA.mp4"
        out_b    = out_dir / f"{stem}_equipoB.mp4"
        out_grid = out_dir / f"{stem}_grid.mp4"

        # Inicializar solo los escritores necesarios según el modo
        writer_main = cv2.VideoWriter(str(tmp_main), fourcc, fps, (canvas_w, canvas_h)) if DO_REID  else None
        writer_a    = cv2.VideoWriter(str(tmp_a),    fourcc, fps, (width, height))       if DO_TEAMS else None
        writer_b    = cv2.VideoWriter(str(tmp_b),    fourcc, fps, (width, height))       if DO_TEAMS else None
        writer_grid = cv2.VideoWriter(str(tmp_grid), fourcc, fps, (grid_w, grid_h))      if DO_GRID  else None

        for w_writer in [writer_main, writer_a, writer_b, writer_grid]:
            if w_writer is not None and not w_writer.isOpened():
                cap.release()
                raise IOError("No se pudo inicializar uno de los VideoWriters.")

        frame_idx = 0
        pbar = tqdm(
            total=total_frames, unit="frame", ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                pbar.update(1)
                timestamp = frame_idx * dt

                # ---- 1. Detección de la Cancha y Calibración de Homografía --------
                pose_results = self.yolo_pose(frame, verbose=False)
                detected_kps = {}
                if len(pose_results) > 0 and pose_results[0].keypoints is not None:
                    kpts = pose_results[0].keypoints
                    xy = kpts.xy.cpu().numpy()[0]
                    conf = kpts.conf.cpu().numpy()[0] if kpts.conf is not None else [1.0] * len(xy)
                    for i in range(min(20, len(xy))):
                        x_kp, y_kp = xy[i]
                        c_kp = conf[i]
                        detected_kps[f"KP{i:02d}"] = (float(x_kp), float(y_kp), float(c_kp))
                
                # Estimar y actualizar homografía (con suavizado EMA temporal)
                self.estimator.estimate(detected_kps)

                # ---- 2. Detección de Jugadores y Balón ---------------------------
                detect_results = self.yolo_detect(frame, conf=min(self.player_conf, self.ball_conf), imgsz=self.imgsz, verbose=False)[0]

                # ---- 3. Extraer y Suavizar Posición de la Pelota -----------------
                raw_ball_pos = None
                best_ball_conf = -1.0
                if len(detect_results.boxes) > 0:
                    boxes = detect_results.boxes
                    cls_arr = boxes.cls.cpu().numpy().astype(int)
                    xyxy_arr = boxes.xyxy.cpu().numpy()
                    conf_arr = boxes.conf.cpu().numpy()
                    
                    for i in range(len(boxes)):
                        if cls_arr[i] == 1:  # Pelota
                            c_ball = conf_arr[i]
                            if c_ball >= self.ball_conf and c_ball > best_ball_conf:
                                best_ball_conf = c_ball
                                x1, y1, x2, y2 = xyxy_arr[i]
                                raw_ball_pos = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                # Actualizar el filtro de Kalman de la pelota
                if raw_ball_pos is not None:
                    if self.ball_filter is None:
                        self.ball_filter = BallKalmanFilter(raw_ball_pos[0], raw_ball_pos[1], dt)
                    else:
                        self.ball_filter.update(raw_ball_pos[0], raw_ball_pos[1])
                    ball_pos = raw_ball_pos
                else:
                    if self.ball_filter is not None:
                        bx, by = self.ball_filter.predict(dt)
                        self.ball_filter.missed_frames += 1
                        if self.ball_filter.missed_frames < 15:
                            ball_pos = (bx, by)
                        else:
                            self.ball_filter = None
                            ball_pos = None
                    else:
                        ball_pos = None

                # ---- 4. Actualizar el Tracking y ReID Multimodal -----------------
                players_to_render = self.tracker.update(frame, detect_results, self.estimator, timestamp)

                # ---- Renderizado Video 1: ReID — Canvas combinado (main) ---------
                canvas_frame = None
                if DO_REID:
                    canvas_frame, mapped_players = self.renderer.render_canvas(
                        frame, players_to_render, ball_pos, self.estimator
                    )
                    self.heatmap_manager.update(mapped_players)
                    writer_main.write(canvas_frame)

                # ---- Renderizado Videos 2 y 3: Análisis por Equipos ---------------
                frame_a = frame_b = None
                if DO_TEAMS:
                    frame_a = TacticalAnalyzer.render_team_focus(
                        frame, players_to_render, ball_pos,
                        focused_class_id=2,
                        class_colors=self.renderer.class_colors,
                        marker_radius=8,
                        show_distances=True
                    )
                    writer_a.write(frame_a)

                    frame_b = TacticalAnalyzer.render_team_focus(
                        frame, players_to_render, ball_pos,
                        focused_class_id=3,
                        class_colors=self.renderer.class_colors,
                        marker_radius=8,
                        show_distances=True
                    )
                    writer_b.write(frame_b)

                # ---- Renderizado Video 4: Cuadrícula 2x2 (solo mode=full) ---------
                if DO_GRID and canvas_frame is not None and frame_a is not None:
                    def _label(img, text, color=(200, 200, 200)):
                        cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
                        cv2.putText(img, text, (10, 21),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)

                    cell_orig = frame.copy()
                    cell_main = cv2.resize(canvas_frame, (width, height))
                    _label(cell_orig, "ORIGINAL")
                    _label(cell_main, "REID + MAPA 2D")
                    _label(frame_a,   "ANALISIS EQUIPO A", self.renderer.class_colors[2])
                    _label(frame_b,   "ANALISIS EQUIPO B", self.renderer.class_colors[3])

                    grid_frame = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
                    grid_frame[0:height, 0:width]      = cell_orig
                    grid_frame[0:height, width:grid_w] = cell_main
                    grid_frame[height:grid_h, 0:width]      = frame_a
                    grid_frame[height:grid_h, width:grid_w] = frame_b

                    cv2.line(grid_frame, (width, 0),      (width, grid_h),  (0, 0, 0), 3)
                    cv2.line(grid_frame, (0, height),     (grid_w, height), (0, 0, 0), 3)
                    writer_grid.write(grid_frame)

        finally:
            cap.release()
            if writer_main is not None: writer_main.release()
            if writer_a    is not None: writer_a.release()
            if writer_b    is not None: writer_b.release()
            if writer_grid is not None: writer_grid.release()
            pbar.close()

        # ---- Generar mapas de calor -----------------------------------------------
        try:
            self.heatmap_manager.generate_and_save_heatmaps(
                renderer=self.renderer, map_w=map_w, map_h=height)
        except Exception as e:
            print(f"[WARNING] Mapas de calor: {e}")

        # ---- Re-codificar con FFmpeg H.264 (solo los que se generaron) ------------
        print("\nRe-codificando videos con FFmpeg H.264...")
        jobs = []
        if DO_REID:  jobs.append((tmp_main, out_main, "ReID + Mapa 2D"))
        if DO_TEAMS: jobs.append((tmp_a,    out_a,    "Equipo A"))
        if DO_TEAMS: jobs.append((tmp_b,    out_b,    "Equipo B"))
        if DO_GRID:  jobs.append((tmp_grid, out_grid, "Grid 2x2 (todos)"))

        for tmp, out, label in jobs:
            if tmp.exists():
                try:
                    _encode_h264(tmp, out)
                    sz = out.stat().st_size / (1024 * 1024)
                    print(f"  [OK] {label:22s} -> {out.name}  ({sz:.1f} MB)")
                except Exception as e:
                    print(f"  [WARN] {label}: {e}")
                    tmp.rename(out)

        print("\n[OK] Pipeline completado.")
        print(f"     Videos exportados en: {out_dir}")
        print()
        print("  Archivos generados:")
        for _, out, label in jobs:
            if out.exists():
                sz = out.stat().st_size / (1024 * 1024)
                print(f"    {out.name:35s} {sz:6.1f} MB  [{label}]")

    # ------------------------------------------------------------------ #
    # Mapa de calor de jugador individual                                  #
    # ------------------------------------------------------------------ #
    def _run_player_heatmap(self, input_path: Path, out_dir: Path, stem: str,
                            fps: float, total_frames: int, dt: float,
                            map_w: int, map_h: int, player_id: int):
        """
        Procesa el video acumulando solo las posiciones del jugador con 'player_id'
        y genera un mapa de calor PNG de alta calidad sobre la cancha 2D.
        """
        class_name_map = {0: "Árbitro", 2: "Equipo A", 3: "Equipo B", 5: "Portero"}

        print("=" * 72)
        print(f"  MAPA DE CALOR — Jugador ID {player_id}")
        print("=" * 72)
        print(f"  Video          : {input_path.name}")
        print(f"  Total frames   : {total_frames}")
        print()

        cap = cv2.VideoCapture(str(input_path))
        positions = []        # Posiciones en metros reales
        player_class = None   # Se detecta automáticamente en el primer hit
        frame_idx = 0

        pbar = tqdm(total=total_frames, unit="frame", ncols=80,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                pbar.update(1)
                timestamp = frame_idx * dt

                # Detección de la cancha
                pose_results = self.yolo_pose(frame, verbose=False)
                detected_kps = {}
                if len(pose_results) > 0 and pose_results[0].keypoints is not None:
                    kpts = pose_results[0].keypoints
                    xy   = kpts.xy.cpu().numpy()[0]
                    conf = kpts.conf.cpu().numpy()[0] if kpts.conf is not None else [1.0] * len(xy)
                    for i in range(min(20, len(xy))):
                        detected_kps[f"KP{i:02d}"] = (float(xy[i][0]), float(xy[i][1]), float(conf[i]))
                self.estimator.estimate(detected_kps)

                # Detección de jugadores
                detect_results = self.yolo_detect(
                    frame, conf=self.player_conf, imgsz=self.imgsz, verbose=False
                )[0]

                # Actualizar tracker
                active_tracks = self.tracker.update(frame, detect_results, self.estimator, timestamp)

                # Acumular solo el jugador solicitado
                for t in active_tracks:
                    if t["track_id"] == player_id:
                        positions.append(t["real_pos"])
                        if player_class is None:
                            player_class = t["class_id"]
                        break
        finally:
            cap.release()
            pbar.close()

        n_positions = len(positions)
        print(f"\n  Posiciones acumuladas para ID {player_id}: {n_positions}")

        if n_positions < 10:
            print(f"\n  ⚠️  Muy pocas apariciones ({n_positions}) para el ID {player_id}.")
            print("     Verifica que el ID sea correcto con scan_player_ids() primero.")
            return

        # ── Construir el canvas del mapa de calor ────────────────────────────────
        map_img, _, _, _ = self.renderer.draw_field(map_w, map_h)

        # Convertir metros a píxeles del canvas
        accum = np.zeros((map_h, map_w), dtype=np.float32)
        for X_m, Y_m in positions:
            px, py = self.renderer.real_to_canvas_px(X_m, Y_m, map_w, map_h)
            if 0 <= px < map_w and 0 <= py < map_h:
                accum[py, px] += 1.0

        # Difuminar para el efecto de calor
        blur_r = int(map_w * 0.07)
        if blur_r % 2 == 0:
            blur_r += 1
        density = cv2.GaussianBlur(accum, (blur_r, blur_r), 0)

        max_val = np.max(density)
        if max_val <= 0:
            print("  ⚠️  No se pudo generar el mapa: todas las posiciones fuera del canvas.")
            return
        density_norm = (density / max_val * 255).astype(np.uint8)

        heatmap_color = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)
        mask = (density_norm > 15).astype(np.uint8)

        result = map_img.copy()
        for c in range(3):
            result[:, :, c] = np.where(
                mask == 1,
                cv2.addWeighted(heatmap_color[:, :, c], 0.70, map_img[:, :, c], 0.30, 0),
                map_img[:, :, c]
            )

        # ── Añadir información del jugador en la imagen ───────────────────────────
        clase_label = class_name_map.get(player_class, "Jugador")
        color_jugador = self.renderer.class_colors.get(player_class, (200, 200, 200))
        titulo = f"MAPA DE CALOR — ID {player_id}  ({clase_label})"
        subtitulo = f"{n_positions} registros de posicion"

        # Fondo del título
        cv2.rectangle(result, (0, 0), (map_w, 52), (10, 10, 10), -1)
        cv2.rectangle(result, (0, 0), (map_w, 52), color_jugador, 2)
        cv2.putText(result, titulo,   (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color_jugador, 2, cv2.LINE_AA)
        cv2.putText(result, subtitulo,(14, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        # Marcador en la posición promedio del jugador
        if positions:
            avg_x = float(np.mean([p[0] for p in positions]))
            avg_y = float(np.mean([p[1] for p in positions]))
            ax, ay = self.renderer.real_to_canvas_px(avg_x, avg_y, map_w, map_h)
            cv2.circle(result, (ax, ay), 10, color_jugador, -1, lineType=cv2.LINE_AA)
            cv2.circle(result, (ax, ay), 10, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.putText(result, f"ID {player_id}", (ax - 18, ay - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Guardar la imagen ─────────────────────────────────────────────────────
        out_dir.mkdir(parents=True, exist_ok=True)
        output_file = out_dir / f"{stem}_heatmap_player_{player_id}.png"
        cv2.imwrite(str(output_file), result)

        sz = output_file.stat().st_size / 1024
        print(f"\n  ✅ Mapa de calor guardado: {output_file.name}  ({sz:.0f} KB)")
        print(f"     Zona promedio del jugador: X={avg_x:.1f}m, Y={avg_y:.1f}m")

