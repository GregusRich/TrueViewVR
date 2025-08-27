from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def load_midas(device: str = "cuda", model_type: str = "DPT_Large"):
    """
    model_type options: "DPT_Large" | "DPT_Hybrid" | "MiDaS_small"
    """
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.dpt_transform if model_type in ["DPT_Large", "DPT_Hybrid"] else transforms.small_transform
    return midas, transform

@torch.no_grad()
def predict_depth(midas, transform, img_pil: Image.Image, device: str = "cuda"):
    import numpy as np
    import torch.nn.functional as F

    # MiDaS hub transforms expect an RGB NumPy array
    np_img = np.array(img_pil)  # RGB HxWx3 uint8
    if np_img.ndim == 2:  # grayscale safeguard
        np_img = np.stack([np_img]*3, axis=-1)

    sample = transform(np_img)            # often returns a dict: {"image": tensor}
    if isinstance(sample, dict) and "image" in sample:
        input_batch = sample["image"].to(device)
    else:
        input_batch = sample.to(device)   # fallback if transform returns a tensor

    pred = midas(input_batch)             # shape (1,H',W')
    h, w = img_pil.size[1], img_pil.size[0]
    pred = F.interpolate(pred.unsqueeze(1), size=(h, w),
                         mode="bicubic", align_corners=False).squeeze(1)
    depth_raw = pred[0].detach().cpu().numpy()

    # Robust normalization for visualization [0,1]
    dmin, dmax = np.percentile(depth_raw, [2, 98])
    depth_vis = np.clip((depth_raw - dmin) / (dmax - dmin + 1e-8), 0, 1)
    return depth_raw, depth_vis
