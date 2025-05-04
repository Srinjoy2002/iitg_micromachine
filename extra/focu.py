# focu.py

import os
import glob

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from scipy.ndimage import distance_transform_edt, gaussian_filter
import open3d as o3d

# ---------------------- Segmentation Model ---------------------- #
_model = None
_input_size = None

def load_segmentation_model(path):
    """Load your ResAttUNet model and remember its input size."""
    global _model, _input_size
    _model = load_model(path, compile=False)
    _, h, w, _ = _model.input_shape
    _input_size = (h, w)
    return _model, _input_size

# ---------------------- Masking via Segmentation ---------------------- #
def segment_mask(image, thresh=0.5):
    """Run the loaded model on one image and return a boolean mask."""
    if _model is None or _input_size is None:
        raise RuntimeError("Model not loaded; call load_segmentation_model() first.")
    h0, w0 = image.shape[:2]
    h, w   = _input_size
    roi    = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    inp    = (roi.astype(np.float32) / 255.0)[None, ...]
    pred   = _model.predict(inp, verbose=0)[0]
    prob   = pred[...,0] if pred.ndim == 3 else pred
    m      = (prob > thresh).astype(np.uint8) * 255

    # keep only largest connected component
    cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(m)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(clean, [c], -1, 255, cv2.FILLED)

    return cv2.resize(clean, (w0, h0), interpolation=cv2.INTER_NEAREST) > 0

# ---------------------- Focus Stacking (idx_map) ---------------------- #
def focus_stack(masks, images):
    arr_gray = [
        cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F) * m
        for img, m in zip(images, masks)
    ]
    stacked = np.stack(arr_gray, axis=0)
    idx_map = np.argmax(stacked, axis=0).astype(np.int32)

    # fill holes by nearest
    valid = stacked.max(axis=0) > 0
    dist, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
    filled = idx_map.copy()
    filled[~valid] = idx_map[iy[~valid], ix[~valid]]
    return filled

# ---------------------- Build Point Cloud ---------------------- #
def build_point_cloud(idx_map, images, layer_dist, xy_scale, z_scale, edge_sigma):
    h, w = idx_map.shape
    mask = idx_map >= 0
    depth = idx_map.astype(np.float32) * layer_dist

    # smooth the rim
    k3 = np.ones((3,3), np.uint8)
    interior = cv2.erode(mask.astype(np.uint8), k3).astype(bool)
    rim = mask & ~interior
    depth_s = gaussian_filter(depth, sigma=edge_sigma)
    depth[rim] = depth_s[rim]

    pts, cols = [], []
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            top = idx_map[y, x]
            color = images[top][y, x].astype(np.float32) / 255.0
            for L in range(top + 1):
                pts.append([x*xy_scale, y*xy_scale, L*layer_dist*z_scale])
                cols.append(color)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float32))
    pcd.colors = o3d.utility.Vector3dVector(np.array(cols, dtype=np.float32))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )
    return pcd

# ---------------------- GUI‐CALLABLE API ---------------------- #
def load_images_from_folder(folder_path):
    """Load all .jpg images (sorted) from a folder."""
    files = sorted(glob.glob(os.path.join(folder_path, '*.jpg')))
    if not files:
        raise IOError(f"No JPGs found in {folder_path!r}")
    return [cv2.imread(f) for f in files]

def process_focus_stack(
    images,
    progress_callback=lambda v: None,
    layer_distance=1.0,
    xy_scale=1.0,
    z_scale=1.0,
    edge_sigma=1.5,
    model_path=r"E:\ACS motion Controller\python code\code\ResAttUNet.h5"
):
    """
    Runs segmentation→stacking→3D build.
    Returns (idx_map, (points, colors), None)
    """
    # 1) load your model once
    if model_path:
        load_segmentation_model(model_path)

    # 2) segment each frame, updating progress
    masks = []
    for i, img in enumerate(images):
        masks.append(segment_mask(img, thresh=0.3))
        progress_callback(i+1)

    # 3) focus-stack
    idx_map = focus_stack(masks, images)

    # 4) build the point cloud
    pcd = build_point_cloud(idx_map, images,
                             layer_dist=layer_distance,
                             xy_scale=xy_scale,
                             z_scale=z_scale,
                             edge_sigma=edge_sigma)

    pts  = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    return idx_map, (pts, cols), None
