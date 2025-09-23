import torch
import numpy as np
from PIL import Image

# Use auto-classes to avoid version-specific names
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

def load_dav2(device: str = "cuda", variant: str = "Small-hf"):
    """
    Load Depth Anything V2 via Hugging Face.
    Variants: 'Small-hf', 'Base-hf', 'Large-hf'
    """
    repo = f"depth-anything/Depth-Anything-V2-{variant}"
    processor = AutoImageProcessor.from_pretrained(repo, trust_remote_code=True)
    model = AutoModelForDepthEstimation.from_pretrained(repo)
    if device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, processor, repo

@torch.no_grad()
def predict_depth(model, processor, img_pil: Image.Image, device: str = "cuda"):
    """
    Returns:
      depth_raw: float array (H, W) in arbitrary units (relative, not metric)
      depth_vis: float array (H, W) normalized to [0,1] for visualization
    """
    inputs = processor(images=img_pil, return_tensors="pt")
    if device == "cuda" and torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    outputs = model(**inputs)  # has .predicted_depth
    depth = outputs.predicted_depth  # shape (1, H', W')
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=img_pil.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze(1)[0]

    depth = depth.float().cpu().numpy()
    vis = depth - depth.min()
    vis = vis / (vis.max() + 1e-8)
    return depth, vis
