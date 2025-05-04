import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from PIL import Image

# Cargar modelo solo una vez
@st.cache_resource
def load_model():
    # return YOLO("last.pt")
    return YOLO("best.pt")

model = load_model()

st.title("🎥🐟 Detección de peces desde archivo de vídeo")

if "processed_frames" in st.session_state:
    if st.button("🔄 Cargar otro vídeo"):
        st.session_state.clear()
        st.rerun()

# Subida de archivo
video_file = st.file_uploader("Sube un archivo de vídeo", type=["mp4", "avi", "mov"])

if video_file is not None:
    if "processed_frames" not in st.session_state:
        # Guardar temporalmente el vídeo
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        video_path = tfile.name

        st.write("Procesando vídeo con YOLO...")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        per_frame_counter = st.empty()
        frame_placeholder = st.empty()
        fish_counter = st.empty()
        progress_bar = st.progress(0)

        processed_frames = []
        per_frame_counts = []
        fish_total = 0

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, stream=False, device="cpu")
            annotated = results[0].plot()

            # Robust fish count by class name
            boxes = results[0].boxes
            names = results[0].names
            class_ids = boxes.cls.int().tolist()
            class_names = [names[int(cls_id)] for cls_id in class_ids]
            num_fish = class_names.count("Fish")
            fish_total += num_fish

            # Convertir a RGB y mostrar
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

        # Guardar en session_state para evitar reprocesamiento
        st.session_state.processed_frames = processed_frames
        st.session_state.per_frame_counts = per_frame_counts
        st.session_state.fish_total = fish_total

    # Mostrar resultados
    st.metric("🎣 Total de peces detectados", st.session_state.fish_total)

    st.write("Explora el vídeo procesado frame por frame:")
    frame_slider = st.slider("Mover por los frames", 0, len(st.session_state.processed_frames) - 1, 0)

    st.markdown(f"**🐟 Peces detectados en este frame:** {st.session_state.per_frame_counts[frame_slider]}")
    st.image(st.session_state.processed_frames[frame_slider], caption=f"Frame {frame_slider + 1}", use_container_width=True)
