import cv2
from config.settings import (
    VIDEO_PATH,
    ORIGINAL_CLIPS_DIR,
    START_MINUTE,
    END_MINUTE,
)

def main():

    ORIGINAL_CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RECORTADOR DE VIDEO")
    print("=" * 60)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("[ERROR] No fue posible abrir el video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(START_MINUTE * 60 * fps)
    end_frame = int(END_MINUTE * 60 * fps)

    print(f"FPS: {fps:.2f}")
    print(f"Resolución: {width} x {height}")
    print(f"Frame inicial: {start_frame}")
    print(f"Frame final: {end_frame}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_path = ORIGINAL_CLIPS_DIR / "clip_01.mp4"
    temp_output_path = ORIGINAL_CLIPS_DIR / "clip_01_temp.mp4"

    writer = cv2.VideoWriter(
        str(temp_output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        print("[ERROR] No se pudo crear el VideoWriter.")
        return

    print("\nEscribiendo video...\n")

    frames_written = 0

    while cap.isOpened():

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if current_frame >= end_frame:
            break

        ret, frame = cap.read()

        if not ret:
            break

        writer.write(frame)

        frames_written += 1

        if frames_written % 100 == 0:
            print(f"{frames_written} frames escritos")

    writer.release()
    cap.release()

    print("\n===================================")
    print("[OK] Clip base generado correctamente")
    print(f"Frames escritos: {frames_written}")

    # Re-encode using imageio-ffmpeg for compatibility
    if temp_output_path.exists():
        print("\nRe-codificando video con FFmpeg para asegurar compatibilidad H.264...")
        import subprocess
        import imageio_ffmpeg as im_ffmpeg
        
        ffmpeg_exe = im_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", str(temp_output_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]
        
        try:
            # Run without printing all internal FFmpeg progress to avoid flooding the terminal
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            temp_output_path.unlink()
            print("[OK] Video re-codificado exitosamente!")
        except Exception as e:
            print(f"[WARNING] Error al re-codificar con FFmpeg: {e}")
            # If FFmpeg failed, fall back to using the raw mp4v file as the output
            temp_output_path.rename(output_path)
            print("[WARNING] Se mantuvo el archivo original como fallback.")

    print(f"Archivo final: {output_path}")

    if output_path.exists():
        size = output_path.stat().st_size / (1024 * 1024)
        print(f"Tamaño: {size:.2f} MB")
    else:
        print("[ERROR] El archivo no fue creado.")


if __name__ == "__main__":
    main()