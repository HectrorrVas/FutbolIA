from ultralytics import YOLO
import torch
from pathlib import Path

# Ruta del modelo
MODEL_PATH = Path(r"E:\Futbol2026\model\best.pt")

print("=" * 60)
print("CARGANDO MODELO...")
print("=" * 60)

# Cargar modelo
model = YOLO(MODEL_PATH)

print("\nModelo cargado correctamente.\n")

# Mostrar clases
print("CLASES ENCONTRADAS:")
for idx, name in model.names.items():
    print(f"{idx}: {name}")

print("\n" + "=" * 60)

# Dispositivo
device = "CUDA" if torch.cuda.is_available() else "CPU"
print(f"Dispositivo: {device}")

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

print("=" * 60)