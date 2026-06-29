import cv2
from tqdm import tqdm
from pathlib import Path

from config.settings import (
    VIDEO_PATH,
    ORIGINAL_CLIPS_DIR,
    START_MINUTE,
    END_MINUTE,
    OUTPUT_WIDTH,
    OUTPUT_HEIGHT,
    OUTPUT_FPS,
)


def main():

    print("=" * 70)
    print("PREPROCESADOR DE VIDEO")
    print("=" * 70)

    ORIGINAL_CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("No fue posible abrir el video.")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = int(START_MINUTE * 60 * original_fps)
    end_frame = int(END_MINUTE * 60 * original_fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_path = ORIGINAL_CLIPS_DIR / "clip_01.mp4"
    temp_output_path = ORIGINAL_CLIPS_DIR / "clip_01_temp.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(temp_output_path),
        fourcc,
        OUTPUT_FPS,
        (OUTPUT_WIDTH, OUTPUT_HEIGHT),
    )

    total = end_frame - start_frame

    print(f"\nProcesando {total} frames...\n")

    pbar = tqdm(total=total)

    frame_number = start_frame

    frame_interval = max(1, round(original_fps / OUTPUT_FPS))

    while frame_number < end_frame:

        ret, frame = cap.read()

        if not ret:
            break

        if (frame_number - start_frame) % frame_interval == 0:

            frame = cv2.resize(
                frame,
                (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )

            writer.write(frame)

        frame_number += 1
        pbar.update(1)

    pbar.close()

    cap.release()
    writer.release()

    print("\nClip base generado correctamente.")

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
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            temp_output_path.unlink()
            print("[OK] Video re-codificado exitosamente!")
        except Exception as e:
            print(f"[WARNING] Error al re-codificar con FFmpeg: {e}")
            temp_output_path.rename(output_path)
            print("[WARNING] Se mantuvo el archivo original como fallback.")

    print(f"Archivo final: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()