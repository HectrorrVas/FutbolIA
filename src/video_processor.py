import cv2
from pathlib import Path


class VideoProcessor:

    def __init__(self, video_path):

        self.video_path = Path(video_path)

        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise FileNotFoundError(
                f"No fue posible abrir el video:\n{self.video_path}"
            )

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.frame_number = 0

    def __iter__(self):
        return self

    def __next__(self):

        ret, frame = self.cap.read()

        if not ret:
            raise StopIteration

        self.frame_number += 1

        return self.frame_number, frame

    def info(self):

        print("\nInformación del video")
        print("-" * 40)
        print(f"FPS: {self.fps:.2f}")
        print(f"Resolución: {self.width} x {self.height}")
        print(f"Frames: {self.total_frames}")

    def release(self):
        self.cap.release()