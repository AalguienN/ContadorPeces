from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
import albumentations as A

fish_augs = A.Compose(
    [
        A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.50),
        A.RandomGamma(gamma_limit=(80, 120),              p=0.40),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=35, p=0.30),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=25, val_shift_limit=15, p=0.30),
        A.GaussianBlur(blur_limit=(3, 5), p=0.20),
        A.MotionBlur(blur_limit=3,        p=0.20),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.20),
        A.HorizontalFlip(p=0.50),
        A.VerticalFlip(p=0.20),
        A.Rotate(limit=10, border_mode=0, p=0.30)
    ],
    bbox_params=A.BboxParams(format="yolo",
                             label_fields=["cls"],   # correct key
                             min_visibility=0.05)
)

YOLODataset.albumentations = fish_augs     # or use custom_augs=… if supported

model = YOLO("yolov8s.pt")

model.train(
    data="config.yaml",
    epochs=50,
    imgsz=1280,
    batch=8,
    mosaic=1.0, mixup=0.5,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    cos_lr=True,       # optional
    patience=8         # stop if no val/box_loss improvement for 8 epochs
)
