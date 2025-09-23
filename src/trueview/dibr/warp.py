import numpy as np
import cv2
from typing import Tuple

def depth_to_disparity_calibrated(
    depth_vis: np.ndarray,
    near_pct: float = 0.90,  # percentile of inverse-depth considered "near"
    far_pct:  float = 0.10,  # percentile considered "far"
    near_px:  float = +22.0, # disparity (px) at near_pct
    far_px:   float = 0.0,   # disparity (px) at far_pct
    gamma:    float = 1.0,
) -> np.ndarray:
    """
    Map normalized depth [0,1] -> horizontal disparity (pixels) by anchoring two
    percentiles of inverse-depth to chosen disparities. For LEFT-eye synthesis
    from RIGHT image, positive disparity shifts pixels RIGHT.

    Returns: disparity_px (float32, same HxW)
    """
    d = np.clip(depth_vis.astype(np.float32), 1e-4, 1.0)
    inv = (1.0 / d) ** float(gamma)

    # Normalize inverse depth for stable percentile math
    inv = (inv - inv.min()) / (inv.max() - inv.min() + 1e-8)

    q_near = np.quantile(inv, float(near_pct))
    q_far  = np.quantile(inv, float(far_pct))

    # Avoid divide-by-zero on extremely flat scenes
    denom = (q_near - q_far)
    if abs(denom) < 1e-6:
        return np.full_like(inv, far_px, dtype=np.float32)

    # Linear map: inv==q_far -> far_px, inv==q_near -> near_px
    t = (inv - q_far) / (denom + 1e-8)
    t = np.clip(t, 0.0, 1.0)
    disp = float(far_px) + t * (float(near_px) - float(far_px))
    return disp.astype(np.float32)


def forward_warp_right_to_left_zbuffer(
    right_bgr: np.ndarray,
    disparity_px: np.ndarray,
    depth_vis: np.ndarray,
    splat_radius: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward-warp RIGHT -> LEFT using a Z-buffer built from 'closeness' = 1/depth.
    Produces TRUE disocclusion holes. 'splat_radius' > 0 will slightly expand coverage.

    Returns:
      left_bgr (uint8), hole_mask_u8 (255=hole)
    """
    H, W = right_bgr.shape[:2]
    left = np.zeros_like(right_bgr, dtype=np.uint8)

    # Z-buffer initialised very low so any valid sample overwrites it
    zbuf = np.full((H, W), -1e9, dtype=np.float32)

    # Convert to closeness; higher means nearer (wins in Z test)
    d = np.clip(depth_vis.astype(np.float32), 1e-4, 1.0)
    z_close = 1.0 / d

    xs = np.arange(W, dtype=np.int32)[None, :].repeat(H, axis=0)
    ys = np.arange(H, dtype=np.int32)[:, None].repeat(W, axis=1)

    # Target X in LEFT frame (RIGHT -> LEFT shift is +disp)
    xt = xs + disparity_px.astype(np.float32)
    xti = np.round(xt).astype(np.int32)

    # Optional small splat to reduce pinholes (does not eliminate true holes)
    R = int(splat_radius)
    for dy in range(-R, R + 1):
        yti = ys + dy
        inside_y = (yti >= 0) & (yti < H)
        if not np.any(inside_y):
            continue

        for dx in range(-R, R + 1):
            xtv = xti + dx
            inside_x = (xtv >= 0) & (xtv < W)
            inside = inside_y & inside_x
            if not np.any(inside):
                continue

            # For ties, prefer nearer and slightly prefer central source pixel to avoid speckle
            # Small penalty from |dx|+|dy|
            penalty = (np.abs(dx) + np.abs(dy)) * 1e-3

            # Flatten per-row to vectorize Z test
            for y in range(H):
                row_mask = inside[y]
                if not np.any(row_mask):
                    continue

                x_src = xs[y, row_mask]
                x_tgt = xtv[y, row_mask]
                z_can = z_close[y, row_mask] - penalty  # nearer => larger

                # Z-test
                better = z_can > zbuf[y, x_tgt]
                if not np.any(better):
                    continue

                idx_t = x_tgt[better]
                idx_s = x_src[better]
                zbuf[y, idx_t] = z_can[better]
                left[y, idx_t] = right_bgr[y, idx_s]

    hole_mask = (zbuf < -1e8).astype(np.uint8) * 255
    # Ensure holes are visibly black for debugging & downstream inpaint
    left[hole_mask == 255] = 0
    return left, hole_mask
