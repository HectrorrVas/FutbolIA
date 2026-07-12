import cv2
import numpy as np
import sys
from pathlib import Path
from ultralytics import YOLO

# Agregar la raíz del proyecto al path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.field_model import FIELD_POINTS, HomographyEstimator, draw_pitch_2d, draw_keypoints_status_2d, meters_to_px

def process_single_image(
    image_path: str,
    pose_model_path: str,
    detect_model_path: str,
    output_path: str = "output/tactical_test_result.png",
    min_kp_conf: float = 0.3,
    min_player_conf: float = 0.25
):
    """
    Procesa una imagen estática para detectar la cancha (keypoints), calcular la homografía,
    detectar los jugadores y proyectar sus posiciones sobre un mapa táctico 2D vertical.
    """
    print("=" * 80)
    print("PROCESANDO IMAGEN TÁCTICA CON DETECCIÓN Y PROYECCIÓN 2D VERTICAL")
    print("=" * 80)
    
    # 1. Cargar la imagen
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen desde: {image_path}")
    
    img_h, img_w = img.shape[:2]
    print(f"Imagen cargada: {Path(image_path).name} ({img_w}x{img_h} px)")
    
    # 2. Cargar modelos de YOLO
    print("\n[1/4] Cargando modelos YOLO...")
    try:
        model_pose = YOLO(pose_model_path)
        print(f"  -> Modelo de Pose (Cancha) cargado: {Path(pose_model_path).name}")
    except Exception as e:
        print(f"  [ERROR] No se pudo cargar el modelo de pose: {e}")
        sys.exit(1)
        
    try:
        model_detect = YOLO(detect_model_path)
        print(f"  -> Modelo de Detección (Jugadores) cargado: {Path(detect_model_path).name}")
    except Exception as e:
        print(f"  [ERROR] No se pudo cargar el modelo de detección: {e}")
        sys.exit(1)

    # 3. Detectar keypoints de la cancha y calcular Homografía
    print("\n[2/4] Ejecutando detección de la cancha (YOLO Pose)...")
    pose_results = model_pose(img, verbose=False)
    
    detected_kps = {}
    if len(pose_results) > 0 and pose_results[0].keypoints is not None:
        # Extraer keypoints del primer objeto cancha detectado
        kpts = pose_results[0].keypoints
        xy = kpts.xy.cpu().numpy()[0]  # Coordenadas (N, 2)
        conf = kpts.conf.cpu().numpy()[0] if kpts.conf is not None else [1.0] * len(xy)
        
        # Mapear a nombres de KP00...KP19
        # Nota: La salida correlativa de YOLO Pose de 0 a 19 se mapea a KP00...KP19
        for i in range(min(20, len(xy))):
            x, y = xy[i]
            c = conf[i]
            detected_kps[f"KP{i:02d}"] = (float(x), float(y), float(c))
            
        print(f"  -> Se obtuvieron {len(detected_kps)} keypoints del modelo de pose.")
    else:
        print("  [WARN] No se detectó la cancha o no se encontraron keypoints.")

    # Inicializar estimador y calcular homografía
    estimator = HomographyEstimator(min_confidence=min_kp_conf)
    success, num_inliers = estimator.estimate(detected_kps)
    
    if not success:
        print("  [ERROR] No se pudo estimar la homografía. Se necesitan al menos 4 inliers.")
        print("          Asegúrate de que la cancha esté visible y bien detectada.")
        sys.exit(1)
        
    print(f"  -> Homografía estimada con éxito. Inliers RANSAC: {num_inliers}/20")

    # 4. Detectar jugadores, árbitro y balón
    print("\n[3/4] Detectando jugadores y balón (YOLO Detect)...")
    detect_results = model_detect(img, conf=min_player_conf, verbose=False)
    
    players_data = []
    ball_pos_img = None
    
    # Mapeo de colores BGR según la clase del proyecto
    class_colors = {
        0: (0, 215, 255),    # Arbitro -> Amarillo
        1: (255, 255, 255),  # Balon -> Blanco
        2: (220, 60, 30),    # EquipoA -> Azul
        3: (30, 30, 220),    # EquipoB -> Rojo
        5: (30, 180, 30),    # Portero -> Verde
    }
    
    # Dibujar marcadores en la imagen de video/original para comparar
    annotated_img = img.copy()
    
    if len(detect_results) > 0 and len(detect_results[0].boxes) > 0:
        boxes = detect_results[0].boxes
        cls_arr = boxes.cls.cpu().numpy().astype(int)
        xyxy_arr = boxes.xyxy.cpu().numpy()
        conf_arr = boxes.conf.cpu().numpy()
        
        for i in range(len(boxes)):
            cls_id = cls_arr[i]
            x1, y1, x2, y2 = xyxy_arr[i]
            conf = conf_arr[i]
            
            # Centro inferior para jugadores (los pies en contacto con el suelo)
            cx = (x1 + x2) / 2.0
            cy = y2
            
            # Centro geométrico completo para el balón
            if cls_id == 1:
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                ball_pos_img = (cx, cy)
                # Dibujar balón en imagen original
                cv2.circle(annotated_img, (int(cx), int(cy)), 6, class_colors[1], -1, lineType=cv2.LINE_AA)
                cv2.circle(annotated_img, (int(cx), int(cy)), 6, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            elif cls_id in [0, 2, 3, 5]:
                players_data.append({
                    "class_id": cls_id,
                    "img_pos": (cx, cy),
                    "bbox": (x1, y1, x2, y2)
                })
                # Dibujar círculo en los pies en la imagen original
                color = class_colors.get(cls_id, (150, 150, 150))
                cv2.circle(annotated_img, (int(cx), int(cy)), 8, color, -1, lineType=cv2.LINE_AA)
                cv2.circle(annotated_img, (int(cx), int(cy)), 8, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                # Bounding box discreto
                cv2.rectangle(annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1, lineType=cv2.LINE_AA)

    print(f"  -> Detectados {len(players_data)} jugadores/árbitros y {1 if ball_pos_img else 0} balón.")

    # 5. Proyectar posiciones y generar el Mapa Táctico 2D
    print("\n[4/4] Proyectando coordenadas al plano 2D real vertical...")
    scale = 10.0  # 10 píxeles por metro
    padding = 30
    
    # Dibujar base de la cancha vertical (41m de ancho, 68m de largo)
    pitch_bg, _ = draw_pitch_2d(scale=scale, padding=padding)
    # Dibujar keypoints con estado de homografía
    pitch_tac = draw_keypoints_status_2d(pitch_bg, estimator, scale=scale, padding=padding)
    
    # Proyectar y dibujar jugadores
    for p in players_data:
        img_pos = p["img_pos"]
        cls_id = p["class_id"]
        
        try:
            # Transformación perspectiva de píxeles de imagen a metros reales
            real_pos = estimator.transform_image_to_real([img_pos])[0]
            X_m, Y_m = real_pos[0], real_pos[1]
            
            # Verificar si el jugador está dentro o cerca de la cancha real (vertical: X=[0,41], Y=[0,68])
            if -3.0 <= X_m <= 44.0 and -3.0 <= Y_m <= 71.0:
                # Convertir metros a píxeles en el canvas 2D usando meters_to_px
                px, py = meters_to_px(X_m, Y_m, scale, padding)
                
                # Dibujar en el mapa táctico
                color = class_colors.get(cls_id, (150, 150, 150))
                cv2.circle(pitch_tac, (px, py), 9, color, -1, lineType=cv2.LINE_AA)
                cv2.circle(pitch_tac, (px, py), 9, (0, 0, 0), 1, lineType=cv2.LINE_AA)
                
                # Pequeño texto con el nombre de la clase (o ID táctico)
                label = "EqA" if cls_id == 2 else "EqB" if cls_id == 3 else "Ref" if cls_id == 0 else "GK"
                cv2.putText(pitch_tac, label, (px - 10, py - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        except Exception as e:
            # Si el punto queda fuera del infinito proyectivo o hay error, se ignora
            pass
            
    # Proyectar y dibujar balón
    if ball_pos_img is not None:
        try:
            real_ball = estimator.transform_image_to_real([ball_pos_img])[0]
            bx, by = meters_to_px(real_ball[0], real_ball[1], scale, padding)
            
            cv2.circle(pitch_tac, (bx, by), 6, class_colors[1], -1, lineType=cv2.LINE_AA)
            cv2.circle(pitch_tac, (bx, by), 6, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        except:
            pass

    # 6. Ensamblar visualización final
    # Redimensionar el mapa táctico para que coincida con la altura de la imagen anotada
    pitch_resized = cv2.resize(pitch_tac, (int(pitch_tac.shape[1] * (annotated_img.shape[0] / pitch_tac.shape[0])), annotated_img.shape[0]))
    
    # Crear canvas lado a lado (Side-by-Side)
    composite = np.hstack([annotated_img, pitch_resized])
    
    # Guardar resultado
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, composite)
    
    print("\n" + "=" * 80)
    print(f"[OK] ¡Procesamiento completado con éxito!")
    print(f"     Resultado visual guardado en: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prueba homografía y detección en una imagen estática.")
    parser.add_argument("--image", type=str, required=True, help="Ruta de la imagen/foto de entrada.")
    parser.add_argument("--pose_model", type=str, default="model/best_pose.pt", help="Ruta al modelo YOLO Pose de la cancha.")
    parser.add_argument("--detect_model", type=str, default="model/best.pt", help="Ruta al modelo YOLO de jugadores.")
    parser.add_argument("--output", type=str, default="output/tactical_test_result.png", help="Ruta de salida de la imagen resultante.")
    
    args = parser.parse_args()
    
    process_single_image(
        image_path=args.image,
        pose_model_path=args.pose_model,
        detect_model_path=args.detect_model,
        output_path=args.output
    )
