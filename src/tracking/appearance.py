import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

class AppearanceModel:
    """
    Modelo encargado de la extracción de características visuales (ReID).
    Combina embeddings profundos con histogramas de color en el espacio HSV.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_deep = True
        try:
            # Usar MobileNet V3 Small como extractor de características rápido y ligero (~5MB)
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            self.model = models.mobilenet_v3_small(weights=weights)
            # Reemplazar el clasificador con Identity para extraer embeddings directos (576 dimensiones)
            self.model.classifier = nn.Identity()
            self.model.to(self.device)
            self.model.eval()
            
            self.transform = transforms.Compose([
                transforms.Resize((128, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print("  [AppearanceModel] Extractor MobileNetV3 inicializado con éxito.")
        except Exception as e:
            print(f"  [AppearanceModel] No se pudo cargar el modelo profundo (ReID): {e}. Usando fallback de color.")
            self.use_deep = False

    def get_deep_embedding(self, crop: np.ndarray) -> np.ndarray:
        """
        Extrae un embedding normalizado de 576 dimensiones de la imagen recortada.
        """
        if not self.use_deep or crop is None or crop.size == 0:
            return None
        try:
            # Convertir de BGR a RGB y luego a imagen PIL
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feat = self.model(tensor)
                
            feat = feat.squeeze(0).cpu().numpy()
            norm = np.linalg.norm(feat)
            return feat / norm if norm > 0 else feat
        except Exception:
            return None

    def get_color_histogram(self, crop: np.ndarray) -> np.ndarray:
        """
        Calcula un histograma de color robusto en el espacio HSV para el recorte de imagen.
        """
        if crop is None or crop.size == 0:
            return None
        try:
            # Convertir a HSV
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            
            # Calcular histogramas para canales H (Tonalidad) y S (Saturación)
            h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256])
            
            # Normalizar individualmente
            cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX)
            
            return np.concatenate([h_hist.flatten(), s_hist.flatten()])
        except Exception:
            return None
