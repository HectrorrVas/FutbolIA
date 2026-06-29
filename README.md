# FutbolIA: Pipeline Profesional de Análisis Táctico de Fútbol

Este proyecto es una plataforma avanzada y modular de análisis deportivo de fútbol basada en Inteligencia Artificial (**YOLOv8 + ByteTrack**). Permite procesar videos de fútbol (especialmente tomas aéreas de drones o cámaras tácticas), realizar el tracking continuo de jugadores y balón, y proyectar visualizaciones analíticas avanzadas directamente en el video y en un mapa táctico 2D.

---

## 🎬 Demo en Video

> 📺 **Video de demostración en YouTube:**
> *[Próximamente - enlace pendiente]*

---

## 🚀 Acceso Rápido

| Recurso | Enlace |
|---|---|
| 📓 **Notebook de Google Colab** | [Abrir en Colab](https://colab.research.google.com/drive/1kCLXkpfXf74LGTfW1IkkSQkKvgjfOGU9) |
| 🤖 **Modelo entrenado + Videos** | [Google Drive](https://drive.google.com/drive/folders/15nc9FGu59d2YMkjfg_UXtFtHlaZ9_4ci) |
| 💻 **Código fuente** | [GitHub](https://github.com/HectrorrVas/FutbolIA) |

---

## 📸 Resultados (Grid 2x2)

El pipeline genera automáticamente un video consolidado en cuadrícula sincronizada de 4 paneles:

1. **ORIGINAL**: El metraje de video original sin marcas para una visualización natural.
2. **VISTA GENERAL + MAPA 2D**: El video con círculos de colores integrados junto a un minimapa táctico 2D en tiempo real.
3. **ANÁLISIS EQUIPO A**: Aísla al Equipo A mostrando su bloque defensivo (Convex Hull), su red de coordinación (vecinos cercanos) y las distancias instantáneas en metros entre jugadores. Los rivales se atenúan en opacidad.
4. **ANÁLISIS EQUIPO B**: Mismo análisis anterior enfocado exclusivamente en el Equipo B.

---

## ✨ Características Principales

* **Filtros One Euro**: Eliminación completa del efecto de parpadeo (*flickering* o *jitter*) de los círculos e indicadores visuales de los jugadores, logrando un movimiento 100% fluido.
* **Detección con Confianza Dual**: Optimización específica para detectar el balón (que suele ser pequeño y rápido) utilizando un umbral de confianza ultra-bajo (`0.12`) y discriminando detecciones de jugadores con un umbral estándar (`0.25`).
* **Mallas de Coordinación**: Dibuja en tiempo real las líneas de vecindad de cada equipo calculando la distancia física aproximada entre los jugadores (en metros reales).
* **Bloques Defensivos (Convex Hull)**: Sombreado dinámico semitransparente del espacio poligonal ocupado por cada equipo para evaluar la compresión y expansión del bloque.
* **Mapas de Calor (Heatmaps)**: Generación automática post-partido de mapas de calor de densidad de posicionamiento, tanto por equipo como de forma individual por cada ID de jugador.
* **Soporte para Cancha Personalizada**: Carga un plano cenital personalizado (`config/field.png`) para el renderizado del mapa táctico 2D vertical.

---

## 📁 Estructura del Proyecto

```text
FutbolIA/
├── config/
│   ├── field.png          # Imagen de fondo del campo de fútbol para el mapa 2D
│   └── settings.py        # Configuraciones generales del proyecto
├── model/
│   └── best.pt            # Pesos entrenados de YOLOv8 (descargar desde Google Drive)
├── clips/
│   ├── raw/               # Videos originales a procesar
│   └── processed/         # Videos de salida (grid, individuales, main)
├── output/
│   └── heatmaps/          # Mapas de calor generados automáticamente (.png)
├── src/
│   ├── analytics.py       # Módulos HeatmapManager y TacticalAnalyzer
│   ├── detect_video.py    # Script de ejecución principal del pipeline
│   ├── filters.py         # Implementación del OneEuroFilter2D
│   ├── processor.py       # Orquestador del pipeline de frames y video
│   ├── renderer.py        # Dibujado del canvas compuesto y campo 2D
│   └── tracker.py         # Gestor de inferencia y tracking con ByteTrack
├── .gitignore             # Archivos excluidos de control de versiones
├── requirements.txt       # Dependencias de Python
└── README.md              # Documentación del proyecto
```

---

## 🛠️ Requisitos e Instalación Local

### 1. Clonar el repositorio
```bash
git clone https://github.com/HectrorrVas/FutbolIA.git
cd FutbolIA
```

### 2. Crear un entorno virtual e instalar dependencias
```bash
python -m venv .venv

# Activar en Windows:
.venv\Scripts\activate
# Activar en Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Descargar el modelo entrenado
Descarga la carpeta del modelo desde [Google Drive](https://drive.google.com/drive/folders/15nc9FGu59d2YMkjfg_UXtFtHlaZ9_4ci) y coloca el archivo `best.pt` dentro de la carpeta `model/`.

---

## 🚀 Cómo Ejecutar

```bash
# En Windows (PowerShell):
$env:PYTHONPATH="."
python src/detect_video.py

# En Linux/macOS:
export PYTHONPATH="."
python src/detect_video.py
```

---

## 📊 Integración con Google Colab (Recomendado para GPU)

> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kCLXkpfXf74LGTfW1IkkSQkKvgjfOGU9)

1. Comprime el proyecto desde Windows:
   ```powershell
   Compress-Archive -Path src, config, model, clips, run_colab.ipynb -DestinationPath FutbolIA_colab.zip -Force
   ```
2. Sube el `.zip` a Colab y ejecuta:
   ```python
   !unzip -q FutbolIA_colab.zip -d /content/FutbolIA/
   !PYTHONPATH=/content/FutbolIA python /content/FutbolIA/src/detect_video.py
   ```

---

## 📝 Modelo YOLO — Clases Detectadas

| ID | Clase | Color en video |
|---|---|---|
| 0 | Árbitro | 🟡 Amarillo |
| 1 | Balón | ⚪ Blanco |
| 2 | Equipo A | 🔵 Azul |
| 3 | Equipo B | 🔴 Rojo |
| 5 | Portero | 🟢 Verde |
