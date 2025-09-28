# src/trueview/cleanup/patchmatch.py
from __future__ import annotations
import numpy as np
import cv2

def _valid_mask_from_holes(hole_mask_u8: np.ndarray) -> np.ndarray:
    return (hole_mask_u8 == 0).astype(np.uint8)

def _pad3(img: np.ndarray, r: int) -> np.ndarray:
    return np.pad(img, ((r, r), (r, r), (0, 0)), mode="reflect")

def _pad1(img: np.ndarray, r: int) -> np.ndarray:
    return np.pad(img, ((r, r), (r, r)), mode="reflect")

def _masked_mean(arr: np.ndarray, mask: np.ndarray, default: float = 1e9) -> float:
    """Mean over arr[mask]; returns default if mask empty."""
    m = arr[mask]
    if m.size == 0:
        return default
    return float(m.mean())

def inpaint_patchmatch_depth_guided(
    img_bgr: np.ndarray,
    hole_mask_u8: np.ndarray,     # 255=hole, 0=keep
    depth01: np.ndarray,          # [0,1] (in your DAV2 vis: smaller = farther, larger = nearer)
    patch: int = 7,               # odd
    iters: int = 4,
    depth_w: float = 0.15,        # depth penalty weight (kept; now secondary)
    rand_search_min: int = 2,     # minimal random window in px
    seed: int | None = 1234,
    far_pct: float = 0.30,        # NEW: donors must be <= this depth percentile (far background)
) -> np.ndarray:
    """
    PatchMatch that copies real pixels from the same image into holes.
    NEW: restrict donors to far-background pixels and compute cost on background-only pixels when possible.
    """
    assert img_bgr.dtype == np.uint8
    H, W = img_bgr.shape[:2]
    r = patch // 2

    # Valid (not masked) pixels
    valid = (hole_mask_u8 == 0).astype(np.uint8)
    if valid.sum() == H * W:
        return img_bgr.copy()

    # Precompute padded arrays
    I = _pad3(img_bgr, r).astype(np.int16)
    Df = np.clip(depth01, 0, 1).astype(np.float32)
    D = _pad1((Df * 255).astype(np.uint8), r)

    # ---- BACKGROUND-ONLY donor mask ----
    # Far background = below far_pct percentile in your depth vis (dark wall in your image)
    far_thr = float(np.quantile(Df[valid.astype(bool)], far_pct))
    donor_ok = (valid == 1) & (Df <= far_thr)   # True where we can safely borrow for wall fills

    # Edge emphasis (unchanged)
    boundary = cv2.Canny(valid * 255, 50, 150)
    boundary = cv2.dilate(boundary, np.ones((3, 3), np.uint8), iterations=1)
    boundary_f = cv2.GaussianBlur(boundary.astype(np.float32), (0, 0), 1.0) / 255.0

    rs = np.random.RandomState(seed) if seed is not None else np.random

    ys, xs = np.where(valid == 0)  # hole pixels
    nnf = np.zeros((H, W, 2), dtype=np.int32)

    # Random valid initialization for each hole pixel — but from BACKGROUND-ONLY donors
    bg_yx = np.column_stack(np.where(donor_ok))
    if bg_yx.size == 0:
        # extreme case: no background donors; fall back to old behavior
        bg_yx = np.column_stack(np.where(valid))
    for y, x in zip(ys, xs):
        yy, xx = bg_yx[rs.randint(0, len(bg_yx))]
        nnf[y, x, 0] = int(yy) - y
        nnf[y, x, 1] = int(xx) - x

    def patch_cost(y, x, dy, dx) -> float:
        y0, x0 = y + r, x + r
        yy, xx = y0 + dy, x0 + dx

        p_src = I[y0 - r:y0 + r + 1, x0 - r:x0 + r + 1]
        p_tgt = I[yy - r:yy + r + 1, xx - r:xx + r + 1]

        # Masks: where the source patch has valid pixels, and where donors are background-only
        vm_src = valid[y - r:y + r + 1, x - r:x + r + 1].astype(bool)
        vm_bg  = donor_ok[y - r:y + r + 1, x - r:x + r + 1].astype(bool)

        # Prefer to compare only where the source patch is valid AND background (avoids candle edges).
        vm = vm_src & vm_bg
        if not vm.any():
            # If no bg pixels in source patch neighborhood, fall back to "valid-only"
            vm = vm_src
            if not vm.any():
                return 1e9

        diff = p_src - p_tgt
        l2 = (diff[..., 0] ** 2 + diff[..., 1] ** 2 + diff[..., 2] ** 2).astype(np.float32)
        c_img = _masked_mean(l2, vm, default=1e9)

        pd_src = D[y0 - r:y0 + r + 1, x0 - r:x0 + r + 1].astype(np.int16)
        pd_tgt = D[yy - r:yy + r + 1, xx - r:xx + r + 1].astype(np.int16)
        d2 = (pd_src - pd_tgt) ** 2
        c_depth = _masked_mean(d2.astype(np.float32), vm, default=1e9)

        w = 1.0 + 2.0 * float(boundary_f[y, x])
        return w * (c_img + depth_w * c_depth)

    # Initial cost
    cost = np.full((H, W), 1e9, dtype=np.float32)
    for y, x in zip(ys, xs):
        dy, dx = nnf[y, x]
        cy = np.clip(y + dy, r, H - r - 1)
        cx = np.clip(x + dx, r, W - r - 1)
        if donor_ok[cy, cx]:  # require donor to be background-OK
            cost[y, x] = patch_cost(y, x, cy - y, cx - x)

    # Iterations with propagation + random search
    for it in range(iters):
        y_range = range(H) if (it % 2 == 0) else range(H - 1, -1, -1)
        x_range = range(W) if (it % 2 == 0) else range(W - 1, -1, -1)
        for y in y_range:
            for x in x_range:
                if valid[y, x]:
                    continue

                nbrs = [(y, x - 1), (y - 1, x)] if (it % 2 == 0) else [(y, x + 1), (y + 1, x)]
                for ny, nx in nbrs:
                    if 0 <= ny < H and 0 <= nx < W:
                        dy, dx = nnf[ny, nx]
                        cy = y + dy
                        cx = x + dx
                        if r <= cy < H - r and r <= cx < W - r and donor_ok[cy, cx]:
                            c = patch_cost(y, x, dy, dx)
                            if c < cost[y, x]:
                                cost[y, x] = c
                                nnf[y, x] = (dy, dx)

                # Random search (coarse-to-fine) — only sample from donor_ok
                R = max(H, W)
                while R >= rand_search_min:
                    ry = rs.randint(-R, R + 1)
                    rx = rs.randint(-R, R + 1)
                    cy = np.clip(y + nnf[y, x, 0] + ry, r, H - r - 1)
                    cx = np.clip(x + nnf[y, x, 1] + rx, r, W - r - 1)
                    if donor_ok[cy, cx]:
                        c = patch_cost(y, x, cy - y, cx - x)
                        if c < cost[y, x]:
                            cost[y, x] = c
                            nnf[y, x] = (cy - y, cx - x)
                    R //= 2

    # Reconstruct
    out = img_bgr.copy()
    for y, x in zip(ys, xs):
        dy, dx = nnf[y, x]
        out[y, x] = img_bgr[np.clip(y + dy, 0, H - 1), np.clip(x + dx, 0, W - 1)]

    # Light blur only inside the mask
    k = 3
    blur = cv2.GaussianBlur(out, (k, k), 0)
    mask3 = (hole_mask_u8 > 0)[:, :, None]
    out = np.where(mask3, blur, out)
    return out
