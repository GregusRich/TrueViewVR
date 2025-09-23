import numpy as np
import cv2
from .camera import make_intrinsics, pose_rt, project_points

def unproject_depth(depth, K):
    h, w = depth.shape
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    zs = depth.reshape(-1)
    X = (xs.reshape(-1) - K[0,2]) * zs / K[0,0]
    Y = (ys.reshape(-1) - K[1,2]) * zs / K[1,1]
    pts = np.stack([X, Y, zs], 1).astype(np.float32)  # (N,3)
    return pts

def apply_pose(T, X):
    N = X.shape[0]
    Xh = np.concatenate([X, np.ones((N,1), np.float32)], 1)
    Xc = (T @ Xh.T).T[:, :3]
    return Xc

def render_pointcloud_zbuffer(pts_left, colors, K_left, out_h, out_w, splat_radius=1.0):
    img  = np.zeros((out_h, out_w, 3), np.uint8)
    zbuf = np.full((out_h, out_w), np.inf, np.float32)

    pix, z = project_points(K_left, pts_left)
    u = np.round(pix[:,0]).astype(np.int32)
    v = np.round(pix[:,1]).astype(np.int32)

    r = int(max(1, splat_radius))
    for i in range(len(u)):
        if z[i] <= 0:
            continue
        uu, vv = u[i], v[i]
        if uu < -r or vv < -r or uu >= out_w+r or vv >= out_h+r:
            continue
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                if dx*dx + dy*dy > r*r:
                    continue
                x, y = uu + dx, vv + dy
                if 0 <= x < out_w and 0 <= y < out_h:
                    if z[i] < zbuf[y, x]:
                        zbuf[y, x] = z[i]
                        img[y, x] = colors[i]
    hole_mask = (zbuf == np.inf).astype(np.uint8) * 255
    return img, hole_mask

def build_and_render_left(right_bgr, depth_m, hfov_deg=60.0, baseline_m=0.064, splat_radius=1.0):
    h, w = right_bgr.shape[:2]
    K = make_intrinsics(w, h, hfov_deg)
    pts_R = unproject_depth(depth_m, K)
    colors = right_bgr.reshape(-1, 3)

    # Right camera at +B/2, left at -B/2 along X (eye baseline)
    T_right = pose_rt(tx=+baseline_m*0.5)
    T_left  = pose_rt(tx=-baseline_m*0.5)
    T_right_inv = np.linalg.inv(T_right)

    # world = inv(T_right)*X_R, then left coords = T_left*world
    pts_world = apply_pose(T_right_inv, pts_R)
    pts_left  = apply_pose(T_left, pts_world)

    left_img, holes = render_pointcloud_zbuffer(pts_left, colors, K, h, w, splat_radius=splat_radius)
    return left_img, holes
