from typing import Optional
import numpy as np
import cv2
from PIL import Image
import torch

def cleanup_with_controlnet(
    left_draft_bgr: np.ndarray,
    left_depth_vis: np.ndarray,          # [0,1] depth vis from our pipeline
    hole_mask: Optional[np.ndarray] = None,  # uint8 0/255; 255 = needs inpaint
    sd_base: str = "runwayml/stable-diffusion-v1-5",
    controlnet_repo: str = "lllyasviel/sd-controlnet-depth",
    denoise: float = 0.35,
    control_scale: float = 1.5,
    keep_draft: bool = True,
    device: str = "cuda",
) -> np.ndarray:
    """
    Depth-conditioned img2img cleanup under left-eye depth.
    If hole_mask is provided, we only replace those regions (dilated) in the final composite.
    """
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    controlnet = ControlNetModel.from_pretrained(controlnet_repo, torch_dtype=torch_dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(sd_base, controlnet=controlnet, torch_dtype=torch_dtype)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    # Control image from left-eye depth (8-bit RGB)
    depth_img = (np.clip(left_depth_vis * 255.0, 0, 255)).astype(np.uint8)
    control_img = Image.fromarray(cv2.cvtColor(depth_img, cv2.COLOR_GRAY2RGB))

    # Init image (the warped draft)
    init_img = Image.fromarray(cv2.cvtColor(left_draft_bgr, cv2.COLOR_BGR2RGB))

    prompt = "high quality, faithful reconstruction, preserve structure, realistic textures"
    negative = "low quality, artifacts, extra objects, deformed"

    strength = float(np.clip(denoise, 0.05, 0.85))
    guidance_scale = 5.5  # small to avoid drifting

    result = pipe(
        prompt=prompt,
        negative_prompt=negative,
        image=control_img,
        num_inference_steps=25,
        controlnet_conditioning_scale=control_scale,
        guidance_scale=guidance_scale,
        init_image=init_img if keep_draft else None,
        strength=strength if keep_draft else 0.75,
    )
    out_bgr = cv2.cvtColor(np.array(result.images[0]), cv2.COLOR_RGB2BGR)

    # If no mask provided, return the whole cleaned image
    if hole_mask is None:
        return out_bgr

    # Hole-aware composite: replace only where we need new content
    mask = hole_mask.copy().astype(np.uint8)
    # Grow a bit to catch tearing borders; then blur for smooth blend
    mask = cv2.dilate(mask, np.ones((7,7), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (15,15), 3)
    m = (mask.astype(np.float32) / 255.0)[..., None]  # HxWx1 in [0,1]

    comp = (out_bgr.astype(np.float32) * m + left_draft_bgr.astype(np.float32) * (1.0 - m)).astype(np.uint8)
    return comp
