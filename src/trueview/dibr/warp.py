from typing import Tuple
import numpy as np
import cv2


def depth_to_disparity(depth_vis: np.ndarray, baseline_px: float = 12.0, gamma: float = 1.0, conv: float = 0.5) -> np.ndarray:
    """
    Convert normalized depth [0,1] to horizontal disparity (pixels) for Left-eye synthesis.
    We use inverse depth so nearer content shifts more. 'conv' sets the convergence plane.
    """
    inv = 1.0 / np.clip(depth_vis, 1e-4, 1.0)
    inv = inv ** gamma
    inv = (inv - inv.min()) / (inv.max() - inv.min() + 1e-8)

    target = np.quantile(inv, conv)   # convergence plane in inv-depth space
    disp = inv - target
    disp = disp / (np.max(np.abs(disp)) + 1e-8)   # [-1,1]
    disp_px = disp * baseline_px
    return disp_px

def depth_to_disparity_calibrated(
    depth_vis: np.ndarray,
    near_pct: float = 0.90,  # percentile of inverse-depth considered "near"
    far_pct:  float = 0.10,  # percentile considered "far"
    near_px:  float = +22.0, # disparity (px) at near_pct
    far_px:   float = 0.0,   # disparity (px) at far_pct
    gamma:    float = 1.0,
):
    """
    Force sensitivity by anchoring two percentiles of inverse-depth to chosen disparities.
    Everything in between is linearly interpolated. Removes the 'zero because close enough'
    behavior you noticed.
    """
    inv = 1.0 / np.clip(depth_vis, 1e-4, 1.0)
    inv = inv ** gamma
    inv = (inv - inv.min()) / (inv.max() - inv.min() + 1e-8)

    inv_far  = np.quantile(inv, far_pct)
    inv_near = np.quantile(inv, near_pct)
    if inv_near <= inv_far:
        inv_near, inv_far = inv_far, inv_near

    disp_px = np.interp(inv, [inv_far, inv_near], [far_px, near_px]).astype(np.float32)

    # comfort clamp
    max_abs = max(abs(near_px), abs(far_px))
    disp_px = np.clip(disp_px, -max_abs, +max_abs)
    return disp_px


def warp_right_to_left(right_bgr: np.ndarray, disparity_px: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warp right image into a left-eye 'draft' using per-pixel horizontal shifts.
    Returns (left_draft_bgr, hole_mask_u8).
    """
    h, w = right_bgr.shape[:2]
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    # For left-eye synthesis, shift pixels to the RIGHT by disparity
    map_x = (x_coords + disparity_px).astype(np.float32)
    map_y = y_coords.astype(np.float32)

    left_draft = cv2.remap(right_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    ones = np.ones((h, w), dtype=np.uint8) * 255
    valid = cv2.remap(ones, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    hole_mask = cv2.bitwise_not(valid)  # 255 where new content is needed
    return left_draft, hole_mask

def warp_depth_right_to_left(depth_vis: np.ndarray, disparity_px: np.ndarray):
    """
    Warp the RIGHT-eye normalized depth [0,1] into the LEFT-eye viewpoint
    using the same backward mapping as the image warp.
    Returns: left_depth_vis (float32 [0,1]), hole_mask (uint8 0/255)
    """
    h, w = depth_vis.shape[:2]
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x_coords + disparity_px).astype(np.float32)
    map_y = y_coords.astype(np.float32)

    left_depth = cv2.remap(
        depth_vis.astype(np.float32), map_x, map_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )

    ones = np.ones((h, w), dtype=np.uint8) * 255
    valid = cv2.remap(
        ones, map_x, map_y, interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    holes = cv2.bitwise_not(valid)
    return left_depth, holes


def inpaint_holes(left_draft_bgr, hole_mask_u8, radius=3, dilate=3, method="telea"):
    mask = hole_mask_u8.copy().astype(np.uint8)
    if dilate > 0:
        mask = cv2.dilate(mask, np.ones((dilate, dilate), np.uint8), iterations=1)
    flag = cv2.INPAINT_NS if method.lower()=="ns" else cv2.INPAINT_TELEA
    return cv2.inpaint(left_draft_bgr, mask, radius, flag)

def make_seam_mask(hole_mask_u8, right_bgr, left_draft_bgr, band_px=8, diff_thresh=10):
    band = cv2.dilate(hole_mask_u8, np.ones((band_px*2+1, band_px*2+1), np.uint8), iterations=1)
    absdiff = cv2.cvtColor(cv2.absdiff(right_bgr, left_draft_bgr), cv2.COLOR_BGR2GRAY)
    seam_like = (absdiff > diff_thresh).astype(np.uint8) * 255
    return cv2.bitwise_or(hole_mask_u8, cv2.bitwise_and(band, seam_like))

def forward_warp_right_to_left_zbuffer(right_bgr: np.ndarray,
                                       disparity_px: np.ndarray,
                                       depth_vis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward warp with Z-buffer compositing.
    - right_bgr: HxWx3 uint8 (RIGHT view)
    - disparity_px: HxW float32, shift >0 moves pixel rightward in LEFT image
    - depth_vis: HxW float32 in [0,1] (we use inverse-depth as 'z' closeness)
    Returns: (left_draft_bgr, hole_mask_u8)
    """
    import numpy as np
    import cv2

    h, w = right_bgr.shape[:2]
    left = np.zeros_like(right_bgr, dtype=np.uint8)
    zbuf = np.full((h, w), -1e9, dtype=np.float32)  # very far

    inv = 1.0 / np.clip(depth_vis.astype(np.float32), 1e-4, 1.0)  # larger = closer

    for y in range(h):
        x_src = np.arange(w, dtype=np.int32)
        disp = disparity_px[y].astype(np.float32)
        x_t = x_src + disp  # target x (float)

        x0 = np.floor(x_t).astype(np.int32)
        a  = (x_t - x0).astype(np.float32)  # subpixel

        # splat into two neighbors for mild antialias
        for s, wgt in ((0, 1.0 - a), (1, a)):
            xt = x0 + s
            valid = (xt >= 0) & (xt < w)
            if not np.any(valid):
                continue
            xtv = xt[valid]
            xsv = x_src[valid]
            z   = inv[y, valid] * wgt[valid]  # weighted z to prefer the nearer contributor

            # update z-buffer: keep nearer (larger 'z')
            better = z > zbuf[y, xtv]
            if not np.any(better):
                continue
            idx = xtv[better]
            src = xsv[better]
            zbuf[y, idx] = z[better]
            left[y, idx] = right_bgr[y, src]

    # holes are where nothing was written
    hole_mask = (zbuf < -1e8).astype(np.uint8) * 255
    return left, hole_mask

