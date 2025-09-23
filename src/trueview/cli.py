# src/trueview/cli.py
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch

# --- Depth (DAV2) ---
from trueview.depth.dav2_infer import load_dav2, predict_depth as dav2_predict

# --- DIBR / warp primitives ---
try:
    from trueview.dibr.warp import (
        depth_to_disparity_calibrated,
        forward_warp_right_to_left_zbuffer,
    )
except Exception as e:
    print("[FATAL] Could not import warp utilities. Make sure warp.py exports "
          "'depth_to_disparity_calibrated' and 'forward_warp_right_to_left_zbuffer'.")
    raise

# --- Optional: depth edge refinement ---
try:
    from trueview.depth.postprocess import refine_depth_edge_aware
    HAS_REFINE = True
except Exception as e:
    print(f"[WARN] depth refinement unavailable: {e}")
    HAS_REFINE = False

# --- Optional ControlNet (kept for the explicit controlnet mode) ---
try:
    # prefer the multi-controlnet version if you added it; otherwise fall back
    from trueview.cleanup.sd_controlnet_inpaint import inpaint_with_multi_controlnet_depth_tile as CN_INPAINT
    CN_MODE = "multi"
except Exception:
    try:
        from trueview.cleanup.sd_controlnet_inpaint import inpaint_with_controlnet_depth as CN_INPAINT
        CN_MODE = "single"
    except Exception as e:
        print(f"[WARN] ControlNet inpaint unavailable: {e}")
        CN_INPAINT = None
        CN_MODE = "none"


# --------------------------- small utilities ---------------------------

def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

def side_by_side(r_bgr: np.ndarray, l_bgr: np.ndarray, out_h: int | None = None) -> np.ndarray:
    if out_h:
        scale = out_h / r_bgr.shape[0]
        r_bgr = cv2.resize(r_bgr, (int(r_bgr.shape[1] * scale), out_h), interpolation=cv2.INTER_AREA)
        l_bgr = cv2.resize(l_bgr, (int(l_bgr.shape[1] * scale), out_h), interpolation=cv2.INTER_AREA)
    # LEFT | RIGHT (so it previews like a stereo pair)
    return np.hstack([l_bgr, r_bgr])

def robust_inpaint_mask(hole_mask_u8: np.ndarray, grow: int = 2, close: int = 3) -> np.ndarray:
    """Dilate + close to catch subpixel seams; returns 255=paint."""
    m = hole_mask_u8.copy().astype(np.uint8)
    if grow > 0:
        m = cv2.dilate(m, np.ones((grow, grow), np.uint8), iterations=1)
    if close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    return (m > 0).astype(np.uint8) * 255

def split_small_large_components(mask_u8: np.ndarray, area_thresh: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Return (small_mask, large_mask) by connected component area; masks are 255=paint."""
    m = (mask_u8 > 0).astype(np.uint8)
    if m.sum() == 0:
        return m, m
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    small = np.zeros_like(m, dtype=np.uint8)
    large = np.zeros_like(m, dtype=np.uint8)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < area_thresh:
            small[labels == i] = 255
        else:
            large[labels == i] = 255
    return small, large

def forward_warp_depth_right_to_left_zbuffer(depth_vis01: np.ndarray, disparity_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward-warp a single-channel depth visualization [0,1] using the same Z-buffer rule as RGB.
    Returns (left_depth_vis01, hole_mask_u8[255=hole]).
    """
    H, W = depth_vis01.shape[:2]
    d3 = np.repeat((depth_vis01 * 255.0).astype(np.uint8)[..., None], 3, axis=2)
    left_d3, hole_mask = forward_warp_right_to_left_zbuffer(d3, disparity_px, depth_vis01, splat_radius=0)
    left_depth = left_d3[..., 0].astype(np.float32) / 255.0
    return left_depth, hole_mask

def save_gray(path: Path, gray01: np.ndarray) -> None:
    cv2.imwrite(str(path), (np.clip(gray01, 0, 1) * 255).astype(np.uint8))


# --------------------------- core processing ---------------------------

def process_frame_right_to_left(
    right_bgr: np.ndarray,
    dav2_model,
    dav2_processor,
    device: str,
    args: argparse.Namespace,
    save_debug_prefix: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RIGHT frame -> LEFT frame
      - DAV2 depth
      - Calibrated disparity
      - Forward Z-buffer -> left draft + holes
      - Robust mask
      - FILL:
          auto:    PatchMatch first, then SDXL for large holes
          patchmatch: PatchMatch only
          sdxl:    SDXL on the whole mask
          controlnet: ControlNet (depth/tile if available)
    Returns: (left_draft_bgr, paint_mask_u8, left_filled_bgr)
    """
    # 1) Depth
    img_pil = Image.fromarray(cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB))
    depth_raw, depth_vis = dav2_predict(dav2_model, dav2_processor, img_pil, device=device)
    if args.refine_depth and HAS_REFINE:
        depth_vis = refine_depth_edge_aware(depth_vis, right_bgr, radius=8, iters=1)
    if save_debug_prefix is not None:
        cv2.imwrite(str(save_debug_prefix.with_name(save_debug_prefix.name + "_depth.png")),
                    (depth_vis * 255).astype(np.uint8))

    # 2) Disparity
    disparity_px = depth_to_disparity_calibrated(
        depth_vis,
        near_pct=args.near_pct, far_pct=args.far_pct,
        near_px=args.near_disp, far_px=args.far_disp,
        gamma=args.gamma,
    )
    if save_debug_prefix is not None:
        disp = (disparity_px - disparity_px.min()) / (disparity_px.max() - disparity_px.min() + 1e-8)
        cv2.imwrite(str(save_debug_prefix.with_name(save_debug_prefix.name + "_disparity.png")),
                    (disp * 255).astype(np.uint8))

    # 3) Forward warp (splat=1 reduces micro-cracks)
    left_draft_bgr, hole_mask = forward_warp_right_to_left_zbuffer(
        right_bgr, disparity_px, depth_vis, splat_radius=1
    )
    if save_debug_prefix is not None:
        cv2.imwrite(str(save_debug_prefix.with_name(save_debug_prefix.name + "_left_draft.png")), left_draft_bgr)
        cv2.imwrite(str(save_debug_prefix.with_name(save_debug_prefix.name + "_left_holes.png")), hole_mask)

    # 4) (Optional) left depth for diagnostics/future guidance
    left_depth_vis, _ = forward_warp_depth_right_to_left_zbuffer(depth_vis, disparity_px)
    if save_debug_prefix is not None:
        cv2.imwrite(str(save_debug_prefix.with_name(save_debug_prefix.name + "_left_depth.png")),
                    (left_depth_vis * 255).astype(np.uint8))

    # 5) Robustify paint mask from true holes
    seam_mask = robust_inpaint_mask(hole_mask, grow=args.mask_grow, close=args.mask_close)
    if save_debug_prefix is not None:
        cv2.imwrite(str(save_debug_prefix.with_name(save_debug_prefix.name + "_mask_final.png")), seam_mask)

    # 6) Choose fill path
    fill_mode = args.fill.lower()

    if fill_mode in ("auto", "patchmatch"):
        # Non-ML first: PatchMatch (depth-guided) for best realism
        try:
            from trueview.cleanup.patchmatch import inpaint_patchmatch_depth_guided
        except Exception as e:
            raise RuntimeError("PatchMatch module missing. Add 'src/trueview/cleanup/patchmatch.py' "
                               "from the previous message.") from e

        pm_out = inpaint_patchmatch_depth_guided(
            left_draft_bgr, seam_mask, left_depth_vis,
            patch=args.pm_patch, iters=args.pm_iters, depth_w=args.pm_depth_w,
            rand_search_min=max(2, args.pm_rand_min), seed=args.seed
        )
        if save_debug_prefix is not None:
            cv2.imwrite(str(save_debug_prefix.with_name(save_debug_prefix.name + "_left_pm.png")), pm_out)

        if fill_mode == "patchmatch":
            # done
            if save_debug_prefix is not None:
                cv2.imwrite(str(save_debug_prefix.with_name(save_debug_prefix.name + "_left_filled.png")), pm_out)
            return left_draft_bgr, seam_mask, pm_out

        # auto: refine big holes with SDXL
        big_mask = split_small_large_components(seam_mask, area_thresh=args.large_area_px)[1]
        if (big_mask > 0).sum() == 0:
            if save_debug_prefix is not None:
                cv2.imwrite(str(save_debug_prefix.with_name(save_debug_prefix.name + "_left_filled.png")), pm_out)
            return left_draft_bgr, seam_mask, pm_out

        # SDXL inpaint (low strength) on big holes only
        try:
            from trueview.cleanup.sdxl_inpaint import inpaint_with_sdxl
        except Exception as e:
            raise RuntimeError("SDXL inpaint module missing. Add 'src/trueview/cleanup/sdxl_inpaint.py' "
                               "from the previous message.") from e

        final = inpaint_with_sdxl(
            pm_out, big_mask,
            sd_xl_repo=args.sdxl_repo,
            steps=args.sdxl_steps, guidance=args.sdxl_guidance, strength=args.sdxl_strength,
            prompt=args.cn_prompt, negative_prompt=args.cn_neg,
            device=device, seed=args.seed
        )
        if save_debug_prefix is not None:
            cv2.imwrite(str(save_debug_prefix.with_name(save_debug_prefix.name + "_left_filled.png")), final)
        return left_draft_bgr, seam_mask, final

    elif fill_mode == "sdxl":
        # SDXL on all masked holes
        try:
            from trueview.cleanup.sdxl_inpaint import inpaint_with_sdxl
        except Exception as e:
            raise RuntimeError("SDXL inpaint module missing. Add 'src/trueview/cleanup/sdxl_inpaint.py'.") from e

        final = inpaint_with_sdxl(
            left_draft_bgr, seam_mask,
            sd_xl_repo=args.sdxl_repo,
            steps=args.sdxl_steps, guidance=args.sdxl_guidance, strength=args.sdxl_strength,
            prompt=args.cn_prompt, negative_prompt=args.cn_neg,
            device=device, seed=args.seed
        )
        if save_debug_prefix is not None:
            cv2.imwrite(str(save_debug_prefix.with_name(save_debug_prefix.name + "_left_filled.png")), final)
        return left_draft_bgr, seam_mask, final

    elif fill_mode == "controlnet":
        if CN_INPAINT is None:
            raise RuntimeError("--fill controlnet requested, but ControlNet module is unavailable.")
        # For multi-controlnet: depth+tile; for single: depth-only
        kwargs = {}
        if CN_MODE == "multi":
            # names in our suggested function
            kwargs.update({
                "sd_base": args.sd_base,
                "depth_repo": args.cn_depth_repo,
                "tile_repo":  args.cn_tile_repo,
                "steps": args.cn_steps,
                "guidance": args.cn_guidance,
                "strength": args.cn_strength,
                "depth_scale": args.cn_depth_scale,
                "tile_scale": args.cn_tile_scale,
            })
        else:
            # single depth controlnet
            kwargs.update({
                "sd_base": args.sd_base,
                "controlnet_repo": args.cn_depth_repo,
                "cn_steps": args.cn_steps,
                "cn_guidance": args.cn_guidance,
                "cn_strength": args.cn_strength,
                "cn_scale": args.cn_depth_scale,
            })

        final = CN_INPAINT(
            left_draft_bgr=left_draft_bgr,
            left_depth_vis=left_depth_vis,
            mask_u8=seam_mask,
            device=device,
            prompt=args.cn_prompt,
            negative_prompt=args.cn_neg,
            seed=args.seed,
            **kwargs,
        )
        if save_debug_prefix is not None:
            cv2.imwrite(str(save_debug_prefix.with_name(save_debug_prefix.name + "_left_filled.png")), final)
        return left_draft_bgr, seam_mask, final

    else:
        raise ValueError(f"Unknown --fill '{fill_mode}'")


# --------------------------- CLI ---------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="TrueViewVR — 2D→Stereo via DAV2 + forward Z-buffer + PatchMatch/SDXL/ControlNet fills"
    )

    ap.add_argument("--input", required=True, help="Path to RIGHT-eye source image or video.")
    ap.add_argument("--outdir", default="data/output", help="Output directory.")
    ap.add_argument("--prefix", default=None, help="Output file prefix (defaults to input stem).")

    # Depth (DAV2)
    ap.add_argument("--dav2-variant", choices=["Small-hf", "Base-hf", "Large-hf"], default="Base-hf")

    # Calibrated disparity
    ap.add_argument("--near-pct", type=float, default=0.90)
    ap.add_argument("--far-pct", type=float, default=0.10)
    ap.add_argument("--near-disp", type=float, default=22.0)
    ap.add_argument("--far-disp", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=1.0)

    # Refinement
    ap.add_argument("--refine-depth", action="store_true")

    # Which fill strategy
    ap.add_argument("--fill", choices=["auto", "patchmatch", "sdxl", "controlnet"], default="auto",
                    help="auto: PatchMatch first, SDXL for large holes; or choose a specific method.")

    # PatchMatch knobs
    ap.add_argument("--pm-patch", type=int, default=7)
    ap.add_argument("--pm-iters", type=int, default=4)
    ap.add_argument("--pm-depth-w", type=float, default=0.15)
    ap.add_argument("--pm-rand-min", type=int, default=2)
    ap.add_argument("--large-area-px", type=int, default=4000)

    # SDXL knobs
    ap.add_argument("--sdxl-repo", default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    ap.add_argument("--sdxl-steps", type=int, default=20)
    ap.add_argument("--sdxl-guidance", type=float, default=3.5)
    ap.add_argument("--sdxl-strength", type=float, default=0.25)

    # ControlNet knobs (used only if --fill controlnet)
    ap.add_argument("--sd-base", default="stabilityai/stable-diffusion-2-inpainting",
                    help="Base inpaint model (for ControlNet path).")
    ap.add_argument("--cn-depth-repo", default="lllyasviel/sd-controlnet-depth")
    ap.add_argument("--cn-tile-repo",  default="lllyasviel/control_v11f1e_sd15_tile")
    ap.add_argument("--cn-steps", type=int, default=22)
    ap.add_argument("--cn-guidance", type=float, default=5.0)
    ap.add_argument("--cn-strength", type=float, default=0.30)
    ap.add_argument("--cn-depth-scale", type=float, default=1.4)
    ap.add_argument("--cn-tile-scale",  type=float, default=0.6)

    # Prompting shared by SDXL/ControlNet
    ap.add_argument("--cn-prompt", default="high quality realistic photo, seamless continuation of the scene")
    ap.add_argument("--cn-neg", default="text, watermark, logo, lowres, blurry, distorted, deformed, artifacts")

    # Mask robustness
    ap.add_argument("--mask-grow", type=int, default=2)
    ap.add_argument("--mask-close", type=int, default=3)

    # Misc
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sbs-h", type=int, default=900, help="Side-by-side image preview height.")

    # Video options
    ap.add_argument("--video-fps", type=float, default=0.0, help="Override FPS for output video (0=match input).")
    ap.add_argument("--video-skip", type=int, default=1, help="Process every Nth frame (>=1).")
    ap.add_argument("--video-max", type=int, default=0, help="Max frames to process (0=all).")

    return ap


def main():
    ap = build_arg_parser()
    args = ap.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = args.prefix if args.prefix is not None else in_path.stem

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Init] device={device}")

    # Load DAV2 once
    model, processor, repo = load_dav2(device=device, variant=args.dav2_variant)
    print(f"[Depth] backend=DAV2 ({repo}), variant={args.dav2_variant}")

    if not is_video_file(in_path):
        # ---------------- IMAGE MODE ----------------
        right_bgr = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
        if right_bgr is None:
            raise FileNotFoundError(f"Could not read image: {in_path}")

        left_draft, paint_mask, left_filled = process_frame_right_to_left(
            right_bgr, model, processor, device, args,
            save_debug_prefix=(outdir / stem)
        )

        # Save stereo previews
        cv2.imwrite(str(outdir / f"{stem}_stereo_sbs.png"),
                    side_by_side(right_bgr, left_draft, out_h=args.sbs_h))
        cv2.imwrite(str(outdir / f"{stem}_stereo_sbs_filled.png"),
                    side_by_side(right_bgr, left_filled, out_h=args.sbs_h))

        print("Outputs:")
        for name in [
            "depth.png", "disparity.png", "left_draft.png", "left_holes.png",
            "left_depth.png", "mask_final.png", "left_filled.png",
            "stereo_sbs.png", "stereo_sbs_filled.png",
        ]:
            f = outdir / f"{stem}_{name}"
            if f.exists():
                print(" ", f.resolve())

    else:
        # ---------------- VIDEO MODE ----------------
        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {in_path}")

        in_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        out_fps = args.video_fps if args.video_fps > 0 else in_fps

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Writers: produce SBS preview videos (Left|Right)
        sbs_size = (W * 2, H)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw_sbs = cv2.VideoWriter(str(outdir / f"{stem}_sbs.mp4"), fourcc, out_fps, sbs_size, True)
        vw_sbs_filled = cv2.VideoWriter(str(outdir / f"{stem}_sbs_filled.mp4"), fourcc, out_fps, sbs_size, True)

        max_frames = args.video_max if args.video_max > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_idx = 0
        written = 0
        t0 = time.time()
        print(f"[Video] {W}x{H} @ {in_fps:.3f} fps | writing @ {out_fps:.3f} fps")
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if args.video_skip > 1 and (frame_idx % args.video_skip) != 0:
                frame_idx += 1
                continue

            left_draft, paint_mask, left_filled = process_frame_right_to_left(
                frame_bgr, model, processor, device, args,
                save_debug_prefix=None
            )

            sbs = side_by_side(frame_bgr, left_draft, out_h=None)        # keep native H
            sbs_filled = side_by_side(frame_bgr, left_filled, out_h=None)

            vw_sbs.write(sbs)
            vw_sbs_filled.write(sbs_filled)

            written += 1
            frame_idx += 1
            if max_frames and written >= max_frames:
                break

            if (written % 5) == 0:
                dt = time.time() - t0
                print(f"  processed {written} frames (every {args.video_skip}th), elapsed {dt/60.0:.1f} min")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cap.release()
        vw_sbs.release()
        vw_sbs_filled.release()

        print("Outputs:")
        for f in [outdir / f"{stem}_sbs.mp4", outdir / f"{stem}_sbs_filled.mp4"]:
            print(" ", f.resolve())


if __name__ == "__main__":
    sys.exit(main())
