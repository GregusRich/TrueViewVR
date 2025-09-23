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
    depth01: np.ndarray,          # [0,1] smaller≈nearer
    patch: int = 7,               # odd
    iters: int = 4,
    depth_w: float = 0.15,        # depth penalty weight
    rand_search_min: int = 2,     # minimal random window in px
    seed: int | None = 1234,
) -> np.ndarray:
    """
    PatchMatch that copies real pixels from the same image into holes.
    Depth regularizer prefers candidate patches with similar depth to local boundary.
    """
    assert img_bgr.dtype == np.uint8
    H, W = img_bgr.shape[:2]
    r = patch // 2

    valid = _valid_mask_from_holes(hole_mask_u8)
    if valid.sum() == H * W:
        return img_bgr.copy()

    I = _pad3(img_bgr, r).astype(np.int16)  # allow diffs without overflow
    D = _pad1((np.clip(depth01, 0, 1) * 255).astype(np.uint8), r)

    # Boundary emphasis helps preserve structure along edges
    boundary = cv2.Canny(valid * 255, 50, 150)
    boundary = cv2.dilate(boundary, np.ones((3, 3), np.uint8), iterations=1)
    boundary_f = cv2.GaussianBlur(boundary.astype(np.float32), (0, 0), 1.0) / 255.0

    rs = np.random.RandomState(seed) if seed is not None else np.random

    ys, xs = np.where(valid == 0)  # hole pixels
    nnf = np.zeros((H, W, 2), dtype=np.int32)

    # Random valid initialization for each hole pixel
    for y, x in zip(ys, xs):
        tries = 0
        while True:
            yy = rs.randint(r, H - r)
            xx = rs.randint(r, W - r)
            if valid[yy, xx]:
                nnf[y, x, 0] = yy - y
                nnf[y, x, 1] = xx - x
                break
            tries += 1
            if tries > 10000:  # extremely degenerate case
                nnf[y, x] = (0, 0)
                break

    def patch_cost(y, x, dy, dx) -> float:
        """Compute cost comparing neighborhood around (y,x) with candidate at offset (dy,dx)."""
        y0, x0 = y + r, x + r
        yy, xx = y0 + dy, x0 + dx

        p_src = I[y0 - r:y0 + r + 1, x0 - r:x0 + r + 1]  # may include holes
        p_tgt = I[yy - r:yy + r + 1, xx - r:xx + r + 1]

        vm = valid[y - r:y + r + 1, x - r:x + r + 1].astype(bool)
        if not vm.any():
            return 1e9  # nothing to compare against

        # Color L2 over valid positions
        diff = p_src - p_tgt
        l2 = (diff[..., 0] ** 2 + diff[..., 1] ** 2 + diff[..., 2] ** 2).astype(np.float32)
        c_img = _masked_mean(l2, vm, default=1e9)

        # Depth similarity (use the same valid mask)
        pd_src = D[y0 - r:y0 + r + 1, x0 - r:x0 + r + 1].astype(np.int16)
        pd_tgt = D[yy - r:yy + r + 1, xx - r:xx + r + 1].astype(np.int16)
        d2 = (pd_src - pd_tgt) ** 2
        c_depth = _masked_mean(d2.astype(np.float32), vm, default=1e9)

        # Heavier near boundaries to respect structure
        w = 1.0 + 2.0 * float(boundary_f[y, x])
        return w * (c_img + depth_w * c_depth)

    # Evaluate initial costs
    cost = np.full((H, W), 1e9, dtype=np.float32)
    for y, x in zip(ys, xs):
        dy, dx = nnf[y, x]
        yy = np.clip(y + dy, r, H - r - 1)
        xx = np.clip(x + dx, r, W - r - 1)
        if valid[yy, xx]:
            cost[y, x] = patch_cost(y, x, yy - y, xx - x)

    # Iterations with propagation + random search
    for it in range(iters):
        y_range = range(H) if (it % 2 == 0) else range(H - 1, -1, -1)
        x_range = range(W) if (it % 2 == 0) else range(W - 1, -1, -1)

        for y in y_range:
            for x in x_range:
                if valid[y, x]:
                    continue

                # Propagate from neighbors
                nbrs = [(y, x - 1), (y - 1, x)] if (it % 2 == 0) else [(y, x + 1), (y + 1, x)]
                for ny, nx in nbrs:
                    if 0 <= ny < H and 0 <= nx < W:
                        dy, dx = nnf[ny, nx]
                        cy = y + dy
                        cx = x + dx
                        if r <= cy < H - r and r <= cx < W - r and valid[cy, cx]:
                            c = patch_cost(y, x, dy, dx)
                            if c < cost[y, x]:
                                cost[y, x] = c
                                nnf[y, x] = (dy, dx)

                # Random search (coarse-to-fine)
                R = max(H, W)
                while R >= rand_search_min:
                    ry = rs.randint(-R, R + 1)
                    rx = rs.randint(-R, R + 1)
                    cy = np.clip(y + nnf[y, x, 0] + ry, r, H - r - 1)
                    cx = np.clip(x + nnf[y, x, 1] + rx, r, W - r - 1)
                    if valid[cy, cx]:
                        c = patch_cost(y, x, cy - y, cx - x)
                        if c < cost[y, x]:
                            cost[y, x] = c
                            nnf[y, x] = (cy - y, cx - x)
                    R //= 2

    # Reconstruct result by copying best-match pixels
    out = img_bgr.copy()
    for y, x in zip(ys, xs):
        dy, dx = nnf[y, x]
        out[y, x] = img_bgr[np.clip(y + dy, 0, H - 1), np.clip(x + dx, 0, W - 1)]

    # Light blur only inside the mask to hide patch seams
    k = 3
    blur = cv2.GaussianBlur(out, (k, k), 0)
    mask3 = (hole_mask_u8 > 0)[:, :, None]
    out = np.where(mask3, blur, out)
    return out
