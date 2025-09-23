# src/trueview/cleanup/sdxl_inpaint.py
from __future__ import annotations
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionXLInpaintPipeline, DPMSolverMultistepScheduler

def inpaint_with_sdxl(
    left_bgr: np.ndarray,
    mask_u8: np.ndarray,          # 255=paint
    *,
    sd_xl_repo: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    steps: int = 20,
    guidance: float = 3.5,
    strength: float = 0.25,       # LOW strength keeps look faithful
    prompt: str = "",             # keep empty/neutral to avoid drift
    negative_prompt: str = "low quality, artifacts, extra objects",
    device: str = "cuda",
    seed: int | None = 1234,
) -> np.ndarray:
    H, W = left_bgr.shape[:2]
    img = Image.fromarray(left_bgr[:, :, ::-1], mode="RGB")
    msk = Image.fromarray(mask_u8, mode="L")

    W8 = (W + 7) // 8 * 8
    H8 = (H + 7) // 8 * 8
    img8 = img.resize((W8, H8), Image.BICUBIC)
    msk8 = msk.resize((W8, H8), Image.NEAREST)

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(sd_xl_repo, torch_dtype=torch.float16)
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

    generator = torch.Generator(device=device) if (seed is not None and torch.cuda.is_available()) else None
    if generator is not None:
        generator.manual_seed(int(seed))

    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=img8,
        mask_image=msk8,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        strength=float(strength),
        generator=generator,
    ).images[0]

    out_bgr = np.array(out.resize((W, H), Image.BICUBIC))[:, :, ::-1]
    keep = (mask_u8 == 0)[:, :, None]
    return np.where(keep, left_bgr, out_bgr).astype(np.uint8)
