# Futbol2026: Pipeline Profesional de Análisis Táctico de Fútbol

Este proyecto es una plataforma avanzada y modular de análisis deportivo de fútbol basada en Inteligencia Artificial (**YOLOv8 + ByteTrack**). Permite procesar videos de fútbol (especialmente tomas aéreas de drones o cámaras tácticas), realizar el tracking continuo de jugadores y balón, y proyectar visualizaciones analíticas avanzadas directamente en el video y en un mapa táctico 2D.

---

## 📸 Demostración de Resultados (Grid 2x2)

El pipeline genera automáticamente un video consolidado en cuadrícula sincronizada de 4 paneles:

1. **ORIGINAL**: El metraje de video original sin marcas para una visualización natural.
2. **VISTA GENERAL + MAPA 2D**: El video con círculos de colores integrados junto a un minimapa táctico 2D en tiempo real.
3. **ANÁLISIS EQUIPO A**: Aísla al Equipo A mostrando su bloque defensivo (Convex Hull), su red de pases/coordinación (vecinos cercanos) y las distancias instantáneas en metros entre jugadores. Los rivales se atenúan en opacidad.
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
Futbol2026/
├── config/
│   ├── field.png          # Imagen de fondo del campo de fútbol para el mapa 2D
│   └── settings.py         # Configuraciones generales del proyecto
├── model/
│   └── best.pt             # Pesos entrenados de YOLOv8l (excluido en gitignore)
├── clips/
│   ├── raw/                # Videos originales a procesar
│   └── processed/          # Videos de salida (grid, individuales, main)
├── output/
│   └── heatmaps/           # Mapas de calor generados automáticamente (.png)
├── src/
│   ├── analytics.py        # Módulos HeatmapManager y TacticalAnalyzer
│   ├── detect_video.py     # Script de ejecución principal del pipeline
│   ├── filters.py          # Implementación del OneEuroFilter2D
│   ├── processor.py        # Orquestador del pipeline de frames y video
│   ├── renderer.py         # Dibujado del canvas compuesto y campo 2D
│   └── tracker.py          # Gestor de inferencia y tracking con ByteTrack
├── .gitignore              # Archivos excluidos de control de versiones
├── requirements.txt        # Dependencias de Python
└── README.md               # Documentación del proyecto
```

---

## 🛠️ Requisitos e Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/Futbol2026.git
cd Futbol2026
```

### 2. Crear un entorno virtual e instalar dependencias
```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
# En Windows:
.venv\Scripts\activate
# En Linux/macOS:
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Agregar los pesos del modelo
Crea la carpeta `model/` si no existe y coloca el archivo de pesos entrenados con el nombre `best.pt`.

---

## 🚀 Cómo Ejecutar

Para iniciar el pipeline completo en tu máquina local:

```bash
# Asegúrate de definir el PYTHONPATH para que Python reconozca el directorio src
# En Windows (PowerShell):
$env:PYTHONPATH="."
python src/detect_video.py

# En Linux/macOS:
export PYTHONPATH="."
python src/detect_video.py
```

---

## 📊 Integración con Google Colab

El proyecto es totalmente compatible con Google Colab para procesamiento acelerado por GPU:

1. Comprime las carpetas necesarias:
   ```powershell
   Compress-Archive -Path src, config, model, clips, run_colab.ipynb -DestinationPath Futbol2026_colab.zip -Force
   ```
2. Sube el `.zip` a tu entorno de Colab.
3. Descomprime y ejecuta el script principal:
   ```python
   !unzip -q Futbol2026_colab.zip -d /content/Futbol2026/
   !PYTHONPATH=/content/Futbol2026 python /content/Futbol2026/src/detect_video.py
   ```
