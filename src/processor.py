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

    def process(self, input_video_path, output_video_path):
        """Procesa el video de entrada y genera todos los videos de análisis."""
        input_path  = Path(input_video_path)
        output_path = Path(output_video_path)
        out_dir     = output_path.parent
        stem = input_path.stem

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

        print("=" * 72)
        print("SISTEMA DE ANALISIS TACTICO DE FUTBOL - Futbol2026")
        print("=" * 72)
        print(f"  Entrada          : {input_path.name}")
        print(f"  Resolución       : {width}x{height}  |  FPS: {fps:.2f}  |  Frames: {total_frames}")
        print(f"  Canvas combinado : {canvas_w}x{canvas_h}")
        print(f"  Salida           : {out_dir}")
        print()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Cuadrícula 2x2
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

        writer_main = cv2.VideoWriter(str(tmp_main), fourcc, fps, (canvas_w, canvas_h))
        writer_a    = cv2.VideoWriter(str(tmp_a),    fourcc, fps, (width, height))
        writer_b    = cv2.VideoWriter(str(tmp_b),    fourcc, fps, (width, height))
        writer_grid = cv2.VideoWriter(str(tmp_grid), fourcc, fps, (grid_w, grid_h))

        for w_writer in [writer_main, writer_a, writer_b, writer_grid]:
            if not w_writer.isOpened():
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

                # ---- Renderizado Video 1: Canvas combinado (main) ----------------
                canvas_frame, mapped_players = self.renderer.render_canvas(
                    frame, players_to_render, ball_pos, self.estimator
                )
                self.heatmap_manager.update(mapped_players)
                writer_main.write(canvas_frame)

                # ---- Renderizado Video 2: Equipo A (enfocado) --------------------
                frame_a = TacticalAnalyzer.render_team_focus(
                    frame, players_to_render, ball_pos,
                    focused_class_id=2,
                    class_colors=self.renderer.class_colors,
                    marker_radius=8,
                    show_distances=True
                )
                writer_a.write(frame_a)

                # ---- Renderizado Video 3: Equipo B (enfocado) --------------------
                frame_b = TacticalAnalyzer.render_team_focus(
                    frame, players_to_render, ball_pos,
                    focused_class_id=3,
                    class_colors=self.renderer.class_colors,
                    marker_radius=8,
                    show_distances=True
                )
                writer_b.write(frame_b)

                # ---- Renderizado Video 4: Cuadrícula 2x2 -------------------------
                cell_orig  = frame.copy()
                cell_main  = cv2.resize(canvas_frame, (width, height))
                cell_a     = frame_a
                cell_b     = frame_b

                # Dibujar etiqueta en cada celda
                def _label(img, text, color=(200, 200, 200)):
                    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
                    cv2.putText(img, text, (10, 21),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)

                _label(cell_orig, "ORIGINAL")
                _label(cell_main, "VISTA GENERAL + MAPA")
                _label(cell_a,    "ANALISIS EQUIPO A", self.renderer.class_colors[2])
                _label(cell_b,    "ANALISIS EQUIPO B", self.renderer.class_colors[3])

                # Crear cuadrícula exacta
                grid_frame = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
                grid_frame[0:height, 0:width] = cell_orig
                grid_frame[0:height, width:grid_w] = cell_main
                grid_frame[height:grid_h, 0:width] = cell_a
                grid_frame[height:grid_h, width:grid_w] = cell_b

                # Dibujar líneas divisorias
                cv2.line(grid_frame, (width, 0), (width, grid_h), (0, 0, 0), 3)
                cv2.line(grid_frame, (0, height), (grid_w, height), (0, 0, 0), 3)

                writer_grid.write(grid_frame)

        finally:
            cap.release()
            writer_main.release()
            writer_a.release()
            writer_b.release()
            writer_grid.release()
            pbar.close()

        # ---- Generar mapas de calor -----------------------------------------------
        try:
            self.heatmap_manager.generate_and_save_heatmaps(
                renderer=self.renderer, map_w=map_w, map_h=height)
        except Exception as e:
            print(f"[WARNING] Mapas de calor: {e}")

        # ---- Re-codificar con FFmpeg H.264 -----------------------------------------
        print("\nRe-codificando videos con FFmpeg H.264...")
        jobs = [
            (tmp_main, out_main, "Main (combinado)"),
            (tmp_a,    out_a,    "Equipo A"),
            (tmp_b,    out_b,    "Equipo B"),
            (tmp_grid, out_grid, "Grid 2x2 (todos)"),
        ]

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
