from ultralytics import YOLO
import custom_layers            # registra GAM, SimSPPF, AKConv
from patch_fasg import inject_fasg_layers

# 1) Carga el modelo base con sus pesos pre-entrenados
model = YOLO("yolov8s.pt")

# 2) Aplica el “parche” FASG
model = inject_fasg_layers(model)

# 3) Entrenar con normalidad
model.train(
    data="config.yaml",
    epochs=50,
    imgsz=1280,
    batch=16,
    device=0           # GPU 0
)
