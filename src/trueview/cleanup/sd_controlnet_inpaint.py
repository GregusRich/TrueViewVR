# src/trueview/cleanup/sd_controlnet_inpaint.py
from __future__ import annotations
import numpy as np
from PIL import Image, ImageOps
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetInpaintPipeline,
    DPMSolverMultistepScheduler,
)

def _depth_to_pil_L(depth_vis01: np.ndarray) -> Image.Image:
    d8 = (np.clip(depth_vis01, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(d8, mode="L")


def _np_to_pil_rgb(arr_uint8_bgr: np.ndarray) -> Image.Image:
    # OpenCV BGR -> PIL RGB
    return Image.fromarray(arr_uint8_bgr[:, :, ::-1].copy(), mode="RGB")

def _depth_to_pil_rgb(depth_vis01: np.ndarray) -> Image.Image:
    d = (np.clip(depth_vis01, 0.0, 1.0) * 255.0).astype(np.uint8)
    d3 = np.repeat(d[..., None], 3, axis=2)  # HxWx3
    return Image.fromarray(d3, mode="RGB")

def _resize_to_multiple_of_8(img: Image.Image, target_wh: tuple[int,int]) -> Image.Image:
    return img.resize(target_wh, Image.BICUBIC)

def inpaint_with_controlnet_depth(
    left_draft_bgr: np.ndarray,
    left_depth_vis: np.ndarray,
    mask_u8: np.ndarray,
    sd_base: str = "runwayml/stable-diffusion-v1-5",
    controlnet_repo: str = "lllyasviel/sd-controlnet-depth",
    cn_steps: int = 30,
    cn_guidance: float = 7.0,
    cn_strength: float = 0.85,
    cn_scale: float = 1.5,
    device: str = "cuda",
    prompt: str = (
        "seamless continuation of the provided image in the same style; "
        "match the original colors, textures, lighting and perspective; "
        "preserve all existing objects and structure; fill only masked regions; no new objects"
    ),
    negative_prompt: str = "low quality, artifacts, warped geometry, incorrect perspective, extra objects",
    seed: int | None = 1234,
) -> np.ndarray:
    """
    Depth-guided inpainting for LEFT eye. IMPORTANT: mask_u8 uses 255=PAINT, 0=KEEP.
    The pipeline only modifies masked pixels; everything else remains from left_draft_bgr.
    """
    H, W = left_draft_bgr.shape[:2]
    pil_image = Image.fromarray(left_draft_bgr[:, :, ::-1])  # BGR->RGB
    pil_depth = Image.fromarray((np.clip(left_depth_vis, 0, 1) * 255).astype(np.uint8), mode="L")
    pil_mask  = Image.fromarray(mask_u8, mode="L")  # 255=paint

    controlnet = ControlNetModel.from_pretrained(controlnet_repo, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        sd_base, controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        pipe.to(device)
    else:
        pipe.enable_model_cpu_offload()

    pipe.enable_vae_slicing()

    # Diffusers likes multiples of 8
    W8 = (W + 7) // 8 * 8
    H8 = (H + 7) // 8 * 8
    pil_image_8 = pil_image.resize((W8, H8), Image.BICUBIC)
    pil_depth_8 = pil_depth.resize((W8, H8), Image.BICUBIC)
    pil_mask_8  = pil_mask.resize((W8, H8), Image.NEAREST)

    # CRITICAL: do NOT invert; white==paint
    mask_for_pipe = pil_mask_8

    generator = torch.Generator(device=device) if (seed is not None and torch.cuda.is_available()) else None
    if generator is not None:
        generator.manual_seed(int(seed))

    result_img = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=pil_image_8,
        mask_image=mask_for_pipe,
        control_image=pil_depth_8,
        num_inference_steps=int(cn_steps),
        guidance_scale=float(cn_guidance),
        strength=float(cn_strength),
        controlnet_conditioning_scale=float(cn_scale),
        generator=generator,
    ).images[0]

    # Resize back and composite ONLY on mask==255
    out_rgb = result_img.resize((W, H), Image.BICUBIC)
    out_bgr = np.array(out_rgb)[:, :, ::-1]
    keep = (mask_u8 == 0)[:, :, None]
    final = np.where(keep, left_draft_bgr, out_bgr).astype(np.uint8)
    return final