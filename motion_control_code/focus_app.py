import os
import glob
import threading
import gc
import cv2
import numpy as np
from tkinter import Tk, Label, Button, filedialog, messagebox, Frame, StringVar
import tkinter.ttk as ttk
import open3d as o3d
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from scipy.spatial import cKDTree

# ---------------------- CONFIGURATION ---------------------- #

unet_model = load_model("ResAttUNet.h5", compile=False)

FOCUS_THRESHOLD = 0.1      # minimal laplacian*mask score
LAYER_DISTANCE  = 100       # mm per layer index
XY_SCALE        = 0.05    # mm per pixel in X/Y
Z_SCALE         = 0.005    # mm per layer index in Z
UPSAMP          = 4        # upsampling factor

# ---------------------- HELPERS ---------------------- #

def load_images_from_folder(folder_path):
    files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
    if not files:
        raise FileNotFoundError(f"No .jpg images in {folder_path}")
    return [cv2.imread(f) for f in files]

def get_unet_mask(img):
    h, w = img.shape[:2]
    inp = cv2.resize(img, (256,256))
    x = img_to_array(inp)/255.0
    x = np.expand_dims(x,0)
    pred = unet_model.predict(x, verbose=0)[0,...,0]
    binm = (pred>0.5).astype(np.uint8)
    return cv2.resize(binm, (w,h), interpolation=cv2.INTER_NEAREST)

def compute_focus_and_stack(images, pb, pct_var):
    H, W = images[0].shape[:2]
    N = len(images)
    score_map = np.zeros((N, H, W), dtype=np.float32)

    total_steps = N * 2  # segmentation + laplacian scoring
    pb["maximum"] = total_steps
    pb["value"] = 0

    # 1) segmentation + laplacian
    for i, img in enumerate(images):
        mask = get_unet_mask(img)
        pb.step(1)
        pct_var.set(f"{(pb['value']/total_steps*100):.0f}%")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        lap_norm = lap / (lap.max()+1e-8)
        score_map[i] = lap_norm * (mask.astype(np.float32))
        pb.step(1)
        pct_var.set(f"{(pb['value']/total_steps*100):.0f}%")

    best_idx = np.argmax(score_map, axis=0).astype(np.int32)
    stacked  = np.zeros_like(images[0])

    # build the stacked image
    for y in range(H):
        for x in range(W):
            idx = best_idx[y,x]
            if score_map[idx,y,x] < (FOCUS_THRESHOLD/255.0):
                stacked[y,x] = (0,0,0)
            else:
                stacked[y,x] = images[idx][y,x]

    return stacked, best_idx

def create_depth_map(idx_map):
    return idx_map.astype(np.float32) * LAYER_DISTANCE

# ---------------------- GUI APP ---------------------- #

class PointCloudApp:
    def __init__(self, master):
        self.master = master
        master.title("3D Point Cloud Reconstruction")
        master.geometry("950x720")

        frame = Frame(master); frame.pack(padx=10, pady=10)
        Label(frame, text="3D Reconstruction", font=("Arial",18,"bold")).pack()

        Button(frame, text="Upload Folder", command=self.upload).pack(pady=5)
        self.proc_btn = Button(frame, text="Process", command=self.start, state="disabled")
        self.proc_btn.pack(pady=5)

        self.dim_lbl = Label(frame, text=""); self.dim_lbl.pack(pady=5)

        # Progress bar + percentage label
        progress_frame = Frame(frame)
        progress_frame.pack(pady=5)
        self.pb = ttk.Progressbar(progress_frame, length=500, mode="determinate")
        self.pb.pack(side="left", padx=(0,10))
        self.pct_var = StringVar(value="0%")
        Label(progress_frame, textvariable=self.pct_var, width=4).pack(side="left")

        self.images = []

    def upload(self):
        fld = filedialog.askdirectory()
        if not fld:
            return
        try:
            self.images = load_images_from_folder(fld)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        messagebox.showinfo("Loaded", f"{len(self.images)} images")
        self.proc_btn.config(state="normal")

    def start(self):
        if not self.images:
            return
        # disable button to prevent double-start
        self.proc_btn.config(state="disabled")
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        # 1) Focus-stack with progress
        stacked, idx_map = compute_focus_and_stack(self.images, self.pb, self.pct_var)

        # 2) Dilate + close (20% of progress)
        self.pb["maximum"] = 100
        self.pb["value"] = 0
        self.pct_var.set("0%")
        d_k = np.ones((5,5),np.uint8)
        c_k = np.ones((7,7),np.uint8)

        idx = cv2.dilate(idx_map.astype(np.uint8), d_k, iterations=3)
        self.pb.step(50); self.pct_var.set("50%")

        idx = cv2.morphologyEx(idx, cv2.MORPH_CLOSE, c_k, iterations=2)
        self.pb.step(50); self.pct_var.set("100%")

        # 3) Create depth & fill zeros
        depth = create_depth_map(idx)
        mask0 = (depth==0)
        if mask0.any():
            coords = np.column_stack(np.nonzero(~mask0))
            vals   = depth[~mask0]
            zc     = np.column_stack(np.nonzero(mask0))
            tree   = cKDTree(coords)
            _, nn  = tree.query(zc, k=1)
            depth[mask0] = vals[nn]

        # 4) Slight blur
        depth = cv2.GaussianBlur(depth, (5,5), sigmaX=1e-3)

        # 5) Upsample depth & stacked RGB
        H, W = depth.shape
        HR = (W*UPSAMP, H*UPSAMP)
        depth_hr   = cv2.resize(depth, HR, interpolation=cv2.INTER_LINEAR)
        stacked_hr = cv2.resize(stacked, HR, interpolation=cv2.INTER_LINEAR)

        # vectorized point generation with jitter
        ys, xs = np.mgrid[0:HR[1], 0:HR[0]]
        zs = depth_hr * Z_SCALE
        valid = zs > 0
        xs = xs[valid]; ys = ys[valid]; zs = zs[valid]
        jitter = (np.random.rand(len(xs),2)-0.5)*(XY_SCALE/UPSAMP)
        wx = (xs + jitter[:,0])*(XY_SCALE/UPSAMP)
        wy = (ys + jitter[:,1])*(XY_SCALE/UPSAMP)
        pts = np.stack((wx, wy, zs), axis=1)
        cols = stacked_hr[valid] / 255.0

        # Build & show point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float32))
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float32))

        aabb = pcd.get_axis_aligned_bounding_box()
        L, Wd, Ht = aabb.get_extent()
        self.master.after(0, self.dim_lbl.config,
                          {"text":f"Size (mm): L={L:.1f}, W={Wd:.1f}, H={Ht:.1f}"})

        o3d.visualization.draw_geometries([pcd])
        gc.collect()
        # re-enable button
        self.proc_btn.config(state="normal")

if __name__ == "__main__":
    root = Tk()
    PointCloudApp(root)
    root.mainloop()
