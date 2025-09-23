import numpy as np

def make_intrinsics(w, h, hfov_deg=60.0):
    hfov = np.deg2rad(hfov_deg)
    fx = (w * 0.5) / np.tan(hfov * 0.5)
    fy = fx
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)
    return K

def pose_rt(tx=0.0, ty=0.0, tz=0.0, rx=0.0, ry=0.0, rz=0.0):
    Rx = np.array([[1,0,0],
                   [0,np.cos(rx),-np.sin(rx)],
                   [0,np.sin(rx), np.cos(rx)]], np.float32)
    Ry = np.array([[ np.cos(ry),0,np.sin(ry)],
                   [0,1,0],
                   [-np.sin(ry),0,np.cos(ry)]], np.float32)
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],
                   [np.sin(rz), np.cos(rz),0],
                   [0,0,1]], np.float32)
    R = Rz @ Ry @ Rx
    t = np.array([tx,ty,tz], np.float32).reshape(3,1)
    T = np.eye(4, dtype=np.float32)
    T[:3,:3] = R; T[:3,3:4] = t
    return T

def project_points(K, Xc):
    z = Xc[:,2].clip(1e-6)
    u = (K[0,0]*Xc[:,0]/z) + K[0,2]
    v = (K[1,1]*Xc[:,1]/z) + K[1,2]
    return np.stack([u,v],1), z
