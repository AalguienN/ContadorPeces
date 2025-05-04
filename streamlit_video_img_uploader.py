import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from PIL import Image
import numpy as np

model_choice = st.selectbox("Selecciona el modelo YOLO a usar:", ["v8n.pt", "v8s.pt"])

# Cargar modelo solo una vez y dependiendo del modelo elegido
if "loaded_model" not in st.session_state or st.session_state.model_path != model_choice:
    with st.spinner(f"Cargando modelo {model_choice}..."):
        st.session_state.loaded_model = YOLO(model_choice)
        st.session_state.model_path = model_choice

model = st.session_state.loaded_model

st.title("🎥🐟 Detección de peces")

# 🔄 Botón para reiniciar
if "processed_frames" in st.session_state or "image_result" in st.session_state:
    if st.button("🔄 Cargar otro archivo"):
        st.session_state.clear()
        st.rerun()

# ✅ Modo: imagen o vídeo
modo = st.radio("¿Qué quieres analizar?", ["Vídeo", "Imagen"])

if modo == "Imagen":
    imagen_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if imagen_file is not None:
        image = Image.open(imagen_file).convert("RGB")
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Inference
        results = model.predict(frame, stream=False, device="cpu")
        annotated = results[0].plot()

        # Conteo
        boxes = results[0].boxes
        names = results[0].names
        class_ids = boxes.cls.int().tolist()
        class_names = [names[int(cls_id)] for cls_id in class_ids]
        num_fish = class_names.count("Fish")

        # Mostrar resultados
        st.markdown(f"**🐟 Peces detectados en la imagen:** {num_fish}")
        st.image(annotated, channels="BGR", use_container_width=True)

elif modo == "Vídeo":
    video_file = st.file_uploader("Sube un archivo de vídeo", type=["mp4", "avi", "mov"])
    if video_file is not None:
        if "processed_frames" not in st.session_state:
            # Guardar temporalmente
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            video_path = tfile.name

            st.write("Procesando vídeo con YOLO...")
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # UI
            per_frame_counter = st.empty()
            frame_placeholder = st.empty()
            fish_counter = st.empty()
            progress_bar = st.progress(0)

            # Datos
            processed_frames = []
            per_frame_counts = []
            fish_total = 0

            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(frame, stream=False, device="cpu")
                annotated = results[0].plot()

                boxes = results[0].boxes
                names = results[0].names
                class_ids = boxes.cls.int().tolist()
                class_names = [names[int(cls_id)] for cls_id in class_ids]
                num_fish = class_names.count("Fish")
                fish_total += num_fish

                # Mostrar mientras se procesa
                img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                frame_placeholder.image(pil_img, caption=f"Procesando... Frame {i + 1}", use_container_width=True)
                per_frame_counter.markdown(f"**🐟 Peces detectados en este frame:** {num_fish}")
                fish_counter.metric("🐟 Peces detectados (acumulado)", fish_total)

                processed_frames.append(pil_img)
                per_frame_counts.append(num_fish)

                progress_bar.progress((i + 1) / total_frames)

            cap.release()
            progress_bar.empty()
            st.success("✅ ¡Procesamiento completado!")

            st.session_state.processed_frames = processed_frames
            st.session_state.per_frame_counts = per_frame_counts
            st.session_state.fish_total = fish_total

        # Mostrar resultado
        st.metric("🎣 Total de peces detectados", st.session_state.fish_total)
        st.write("Explora el vídeo procesado frame por frame:")
        frame_slider = st.slider("Mover por los frames", 0, len(st.session_state.processed_frames) - 1, 0)
        st.markdown(f"**🐟 Peces detectados en este frame:** {st.session_state.per_frame_counts[frame_slider]}")
        st.image(st.session_state.processed_frames[frame_slider], caption=f"Frame {frame_slider + 1}", use_container_width=True)
