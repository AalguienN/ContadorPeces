# patch_fasg.py ─ Modifica una instancia de YOLOv8s en memoria
from ultralytics import YOLO
import torch.nn as nn
from ultralytics.nn.modules import SPPF, C2f, Conv
from custom_layers import SimSPPF, GAM, AKConv

def inject_fasg_layers(ymodel: YOLO) -> YOLO:
    """
    Parchea una instancia YOLO (Ultralytics) para:
      1. SPPF  → SimSPPF
      2. Insertar un bloque GAM tras el 1er C2f (backbone)
      3. Reemplazar cada Conv 3×3 stride 1 del *neck* por AKConv
    """
    gam_inserted = False           # nos aseguramos de meter solo UNO

    def _recursive(parent):
        nonlocal gam_inserted
        # usamos list() para evitar problemas al modificar mientras iteramos
        for name, module in list(parent.named_children()):

            # 1) SPPF ➜ SimSPPF
            if isinstance(module, SPPF):
                # obtener kernel size sin asumir tipo
                if hasattr(module, "m"):                       # MaxPool2d
                    ks = module.m.kernel_size
                    ksize = ks[0] if isinstance(ks, (tuple, list)) else ks
                else:
                    ksize = 5                                  # fallback

                new = SimSPPF(module.cv1.conv.in_channels,
                              module.cv2.conv.out_channels,
                              k=ksize)
                new.load_state_dict(module.state_dict(), strict=False)
                setattr(parent, name, new)

            # 2) GAM tras el PRIMER C2f del backbone
            elif isinstance(module, C2f) and not gam_inserted:
                gam = GAM(module.cv2.conv.out_channels)
                setattr(parent, name, nn.Sequential(module, gam))
                gam_inserted = True

            # 3) Conv 3×3 stride 1 ➜ AKConv (solo en el neck)
            elif isinstance(module, Conv):
                k, s = module.conv.kernel_size, module.conv.stride
                # elegimos solo las 3×3 stride 1 -- las del neck cumplen esto
                if k == (3, 3) and s == (1, 1):
                    new = AKConv(module.conv.in_channels,
                                 module.conv.out_channels,
                                 k=3, stride=1,
                                 bias=module.conv.bias is not None)
                    setattr(parent, name, new)

            # Recorremos los hijos (importante: hazlo al final)
            _recursive(module)

    _recursive(ymodel.model)
    return ymodel
