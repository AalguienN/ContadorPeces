# 🐟 Contador de Peces con YOLOv8
Este repositorio forma parte del trabajo final de la asignatura Herramientas y Aplicaciones de la Inteligencia Artificial (HAIA) del Máster en Inteligencia Artificial, Reconocimiento de Formas e Imagen Digital (MIARFID) de la Universitat Politècnica de València (UPV).

Autores:
Adrián Camacho García
Francisco Galindo Santos

# 📌 Descripción del Proyecto
El objetivo de este proyecto es desarrollar un sistema de conteo automático de peces en entornos subacuáticos utilizando visión por computador e inteligencia artificial. Para ello, se ha utilizado un enfoque distribuido con cámaras subacuáticas conectadas a un servidor en la nube (AWS EC2), donde se ejecuta un modelo de detección de objetos basado en la familia YOLOv8.

El sistema permite subir imágenes o vídeos subacuáticos a través de una interfaz web (Streamlit), realizar inferencia sobre los datos y mostrar el número de peces detectados junto con anotaciones visuales.

# ⚙️ Funcionalidades
Entrenamiento de modelos YOLOv8 personalizados con técnicas de fine-tuning.

- Comparativa de tres variantes optimizadas para entornos submarinos:
  - YOLOv8-Muskit-DA (aumentos de datos específicos)
  - YOLOv8-AquaYOLO (mejoras en el backbone y atención contextual)
  - YOLOv8-FASG (convoluciones deformables y atención global)

- Interfaz de usuario con Streamlit para:
  - Subida de imágenes o vídeos.
  - Visualización del conteo de peces.
  - Exploración de detecciones por frame.

Despliegue funcional probado en instancias EC2 (t3.large, CPU-only)

# 🧾 Dataset
Se utilizó una versión adaptada para entrenamiento con YOLO disponible en el repositorio [YOLO-Fish](https://github.com/tamim662/YOLO-Fish). Para realizar los entrenamientos se descargó y se modificó la estructura de carpetas para que fuera compatible con los scripts que proporcionamos.

# 🚀 Despliegue
El sistema fue desplegado en una instancia EC2 de AWS (tipo t3.large, sin GPU) usando Streamlit. La interfaz es ligera y permite la inferencia en CPU en tiempo razonable para redes pequeñas (como YOLOv8n).

descargar `requirements.txt`

Inicializar con comando: `streamlit_video_img_uploader.py`
