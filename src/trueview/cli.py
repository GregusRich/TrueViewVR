# src/trueview/cli.py
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from trueview.depth.postprocess import refine_depth_edge_aware
from trueview.depth.midas_infer import load_midas, predict_depth
from trueview.dibr.warp import protect_foreground_mask


from trueview.dibr.warp import (
    depth_to_disparity,                 # quantile mapping
    depth_to_disparity_calibrated,      # calibrated mapping (make stereo “sensitive”)
    warp_right_to_left,
    warp_depth_right_to_left,
    inpaint_holes,
    make_seam_mask,
)

# Optional: depth edge refinement
try:
    from trueview.depth import refine_depth_edge_aware
    HAS_REFINE = True
except Exception as e:
    print(f"[WARN] depth refinement unavailable: {e}")
    HAS_REFINE = False

# Optional: diffusion cleanup
try:
    from trueview.cleanup.controlnet_cleanup import cleanup_with_controlnet
    HAS_CLEANUP = True
except Exception:
    HAS_CLEANUP = False


def side_by_side(r_bgr, l_bgr, out_h=None):
    if out_h:
        scale = out_h / r_bgr.shape[0]
        r_bgr = cv2.resize(r_bgr, (int(r_bgr.shape[1]*scale), out_h), interpolation=cv2.INTER_AREA)
        l_bgr = cv2.resize(l_bgr, (int(l_bgr.shape[1]*scale), out_h), interpolation=cv2.INTER_AREA)
    return np.hstack([r_bgr, l_bgr])


def main():
    ap = argparse.ArgumentParser(description="TrueViewVR — MiDaS + DIBR single-frame")
    ap.add_argument("--input", required=True, help="Path to the RIGHT-eye source image.")
    ap.add_argument("--outdir", default="data/output", help="Output directory.")

    # Depth & mapping
    ap.add_argument("--midas", default="DPT_Large",
                    choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"], help="MiDaS model.")
    ap.add_argument("--map", choices=["quantile","calib"], default="calib",
                    help="Depth→disparity mapping (calib recommended).")
    # quantile mapping dials
    ap.add_argument("--baseline-px", type=float, default=12.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--conv", type=float, default=0.5)
    # calibrated mapping dials
    ap.add_argument("--near-pct", type=float, default=0.90)
    ap.add_argument("--far-pct",  type=float, default=0.10)
    ap.add_argument("--near-disp", type=float, default=22.0)
    ap.add_argument("--far-disp",  type=float, default=0.0)

    # Refinement & fill
    ap.add_argument("--refine-depth", action="store_true", help="Edge-aware refine of depth (if available).")
    ap.add_argument("--fill", choices=["inpaint","cleanup","none"], default="inpaint",
                    help="Hole/seam fill method.")
    ap.add_argument("--seam-band", type=int, default=10, help="Seam band (px) around holes to replace.")
    ap.add_argument("--seam-thresh", type=int, default=10, help="Absdiff threshold for seam detection.")
    ap.add_argument("--inpaint-radius", type=int, default=3, help="OpenCV inpaint radius.")
    # Cleanup (diffusion) dials
    ap.add_argument("--sd-base", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--controlnet", default="lllyasviel/sd-controlnet-depth")
    ap.add_argument("--denoise", type=float, default=0.35)
    ap.add_argument("--control-scale", type=float, default=1.9)
    ap.add_argument("--keep-draft", action="store_true", help="Use warped draft as init to reduce drift.")

    # Preview
    ap.add_argument("--sbs-h", type=int, default=720)

    # Warp
    ap.add_argument("--warp", choices=["backward", "forward"], default="forward",
                    help="Warp mode: backward remap or forward Z-buffer (recommended).")

    args = ap.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    stem = in_path.stem

    # RIGHT image
    right_bgr = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if right_bgr is None:
        raise FileNotFoundError(f"Could not read image: {in_path}")

    # Depth (MiDaS only)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    midas, transform = load_midas(device=device, model_type=args.midas)
    img_pil = Image.fromarray(cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB))
    depth_raw, depth_vis = predict_depth(midas, transform, img_pil, device=device)

    if args.refine_depth:
        if HAS_REFINE:
            depth_vis = refine_depth_edge_aware(depth_vis, right_bgr, radius=8, iters=1)
            cv2.imwrite(str(outdir / f"{stem}_depth_refined.png"), (depth_vis * 255).astype(np.uint8))
        else:
            print("[WARN] --refine-depth requested but not available; skipping.")

    # Save RIGHT depth
    cv2.imwrite(str(outdir / f"{stem}_depth.png"), (depth_vis*255).astype(np.uint8))

    # Disparity
    if args.map == "calib":
        disparity_px = depth_to_disparity_calibrated(
            depth_vis,
            near_pct=args.near_pct, far_pct=args.far_pct,
            near_px=args.near_disp,  far_px=args.far_disp,
            gamma=args.gamma,
        )
    else:
        disparity_px = depth_to_disparity(
            depth_vis, baseline_px=args.baseline_px, gamma=args.gamma, conv=args.conv
        )

    print(f"Disparity px: min={disparity_px.min():.2f}, max={disparity_px.max():.2f}, mean={disparity_px.mean():.2f}")
    disp_norm = (disparity_px - disparity_px.min()) / (disparity_px.max() - disparity_px.min() + 1e-8)
    cv2.imwrite(str(outdir / f"{stem}_disparity.png"), (disp_norm*255).astype(np.uint8))

    # LEFT depth (warped) + diff
    left_depth_vis, _ = warp_depth_right_to_left(depth_vis, disparity_px)
    cv2.imwrite(str(outdir / f"{stem}_left_depth.png"), (left_depth_vis*255).astype(np.uint8))
    depth_diff = cv2.absdiff((depth_vis*255).astype(np.uint8), (left_depth_vis*255).astype(np.uint8))
    cv2.imwrite(str(outdir / f"{stem}_depth_diff.png"), depth_diff)

    # LEFT draft + hole mask (choose warp mode)
    if args.warp == "forward":
        from trueview.dibr.warp import forward_warp_right_to_left_zbuffer
        left_draft_bgr, hole_mask = forward_warp_right_to_left_zbuffer(
            right_bgr, disparity_px, depth_vis
        )
    else:
        left_draft_bgr, hole_mask = warp_right_to_left(right_bgr, disparity_px)

    cv2.imwrite(str(outdir / f"{stem}_left_draft.png"), left_draft_bgr)
    cv2.imwrite(str(outdir / f"{stem}_left_holes.png"), hole_mask)
    print(f"[Warp] mode={args.warp}, holes={(hole_mask > 0).sum()} px")

    # Seam-aware mask to remove the ghost edge (holes + narrow band)
    # Seam-aware mask = holes + small band around stretched/ghost edges
    seam_mask = make_seam_mask(
        hole_mask, right_bgr, left_draft_bgr,
        band_px=args.seam_band, diff_thresh=args.seam_thresh
    )

    # --- PROTECT FOREGROUND: do NOT repaint near objects (e.g., candle) ---
    # Build a mask of the closest ~14% pixels and dilate slightly.
    fg_keep = protect_foreground_mask(depth_vis, near_pct=0.86, grow=1)
    # Remove foreground from repaint mask
    seam_mask = cv2.bitwise_and(seam_mask, cv2.bitwise_not(fg_keep))

    # (optional debug saves)
    cv2.imwrite(str(outdir / f"{stem}_mask_foreground.png"), fg_keep)
    dbg = right_bgr.copy()
    dbg[seam_mask > 0] = (0.3 * dbg[seam_mask > 0] + 0.7 * np.array([0, 0, 255])).astype(np.uint8)  # red=repaint
    cv2.imwrite(str(outdir / f"{stem}_mask_repaint.png"), dbg)

    # final mask used by inpaint/cleanup
    cv2.imwrite(str(outdir / f"{stem}_mask_final.png"), seam_mask)

    # Fill
    left_filled = left_draft_bgr.copy()
    if args.fill == "inpaint":
        left_filled = inpaint_holes(left_draft_bgr, seam_mask, radius=args.inpaint_radius, dilate=0, method="telea")
        cv2.imwrite(str(outdir / f"{stem}_left_inpaint.png"), left_filled)
    elif args.fill == "cleanup":
        if not HAS_CLEANUP:
            print("[WARN] Cleanup requested but cleanup module not available; falling back to inpaint.")
            left_filled = inpaint_holes(left_draft_bgr, seam_mask, radius=args.inpaint_radius, dilate=0, method="telea")
        else:
            left_filled = cleanup_with_controlnet(
                left_draft_bgr=left_draft_bgr,
                left_depth_vis=left_depth_vis,   # use LEFT-eye depth for conditioning
                hole_mask=seam_mask,             # replace seam too (no double objects)
                sd_base=args.sd_base,
                controlnet_repo=args.controlnet,
                denoise=args.denoise,
                control_scale=args.control_scale,
                keep_draft=args.keep_draft,
                device=device,
            )
            cv2.imwrite(str(outdir / f"{stem}_left_clean.png"), left_filled)

    # Stereo previews
    cv2.imwrite(str(outdir / f"{stem}_stereo_sbs.png"), side_by_side(right_bgr, left_draft_bgr, out_h=args.sbs_h))
    cv2.imwrite(str(outdir / f"{stem}_stereo_sbs_filled.png"), side_by_side(right_bgr, left_filled, out_h=args.sbs_h))

    print("Done:")
    for name in ["depth.png","disparity.png","left_depth.png","depth_diff.png",
                 "left_draft.png","left_holes.png","mask_final.png","stereo_sbs.png","stereo_sbs_filled.png"]:
        print("  ", (outdir / f"{stem}_{name}").resolve())


if __name__ == "__main__":
    main()
