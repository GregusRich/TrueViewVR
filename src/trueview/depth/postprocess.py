import numpy as np
import cv2

def refine_depth_edge_aware(depth_vis: np.ndarray,
                            rgb_bgr: np.ndarray,
                            radius: int = 8,
                            iters: int = 1) -> np.ndarray:
    """
    Dependency-free edge-aware refinement.
    - Smooths depth while respecting image edges (approximate).
    - Uses only standard OpenCV ops; no ximgproc / guided filter required.
    Input/Output: depth_vis in [0,1].
    """
    d = depth_vis.astype(np.float32)
    # soft edge map from the RGB image
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gradx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grady = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gradx, grady)  # larger at edges
    mag = cv2.GaussianBlur(mag, (0, 0), sigmaX=1.2)

    out = d.copy()
    for _ in range(iters):
        # Smooth depth
        out = cv2.bilateralFilter(out, d=0, sigmaColor=0.06 * 255, sigmaSpace=radius)
        # Pull depth towards original near strong edges to preserve boundaries
        w = np.clip(mag * 4.0, 0.0, 1.0)  # weight near edges
        out = (1.0 - w) * out + w * d
    return np.clip(out, 0.0, 1.0)


