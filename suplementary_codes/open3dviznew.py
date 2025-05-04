import os
import glob
import threading

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.ndimage import median_filter, distance_transform_edt, gaussian_filter
from tkinter import Tk, Label, Button, Entry, filedialog, messagebox, Frame
from tkinter import ttk

import open3d as o3d
from matplotlib import cm

# ---------------------- Segmentation Model ---------------------- #
def load_segmentation_model(path):
    model = load_model(path, compile=False)
    _, h, w, _ = model.input_shape
    return model, (h, w)

# ---------------------- Masking via Segmentation ---------------------- #
def segment_mask(model, input_size, image, thresh=0.1):
    h0, w0 = image.shape[:2]
    h, w   = input_size
    roi    = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    inp    = np.expand_dims(roi.astype(np.float32) / 255.0, axis=0)
    pred   = model.predict(inp, verbose=0)[0]
    prob   = pred[...,0] if pred.ndim==3 else pred
    m      = (prob > thresh).astype(np.uint8)*255

    # keep only largest component
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(m)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(clean, [c], -1, 255, cv2.FILLED)
    return cv2.resize(clean, (w0, h0), interpolation=cv2.INTER_NEAREST)

# ---------------------- Focus Stacking ---------------------- #
def focus_stack(images):
    arr = np.stack(images, axis=0)
    n, h, w, _ = arr.shape

    measure = np.zeros((h, w), dtype=np.float32)
    idxs    = -1*np.ones((h, w), dtype=np.int32)

    for i in range(n):
        gray = cv2.cvtColor(arr[i], cv2.COLOR_BGR2GRAY)
        lap  = cv2.Laplacian(gray, cv2.CV_64F)
        mask = lap > measure
        measure[mask] = lap[mask]
        idxs[mask]    = i

    # fill holes by nearest neighbor
    valid = idxs >= 0
    dist, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
    filled = idxs.copy()
    filled[~valid] = idxs[iy[~valid], ix[~valid]]

    # median filter to remove speckles
    idx_map = median_filter(filled, size=3)

    return idx_map

# ---------------------- Open3D Visualization ---------------------- #
def visualize_open3d(idx_map, layer_dist, xy_scale=1.0, z_scale=1.0, gauss_sigma=1):
    # 1) Build continuous depth
    depth = idx_map.astype(np.float32) * layer_dist
    depth = median_filter(depth, size=3)
    depth = gaussian_filter(depth, sigma=gauss_sigma)

    h, w = depth.shape
    mask_any = idx_map >= 0

    # 2) Create point cloud
    xs = np.arange(w) * xy_scale
    ys = np.arange(h) * xy_scale
    Xg, Yg = np.meshgrid(xs, ys)

    zs = depth * z_scale
    valid = mask_any

    pts = np.vstack((Xg[valid], Yg[valid], zs[valid])).T

    # Color by height via HSV colormap
    norm = (zs - np.nanmin(zs)) / (np.nanmax(zs) - np.nanmin(zs) + 1e-8)
    colors = cm.hsv(norm)
    cols = colors[..., :3][valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)

    # 3) Visualize
    o3d.visualization.draw_geometries([pcd], window_name="3D Focus Stack")

# ---------------------- GUI App ---------------------- #
class FocusStackApp:
    def __init__(self, master):
        self.master = master
        master.title('3D Focus Stack (Open3D)')
        master.geometry('800x600')

        self.model = None
        self.input_size = None
        self.images = []
        self.paths  = []

        frm = Frame(master); frm.pack(padx=10, pady=10)
        Button(frm, text='Load .h5 Model', command=self.load_model).pack(pady=5)
        Button(frm, text='Upload Images',   command=self.upload_images).pack(pady=5)

        Label(frm, text='Layer Thickness:').pack(pady=2)
        self.ld = Entry(frm); self.ld.insert(0, '1.0'); self.ld.pack(pady=2)
        Label(frm, text='Edge Smooth σ:').pack(pady=2)
        self.es = Entry(frm); self.es.insert(0, '1');   self.es.pack(pady=2)

        Button(frm, text='Stack & Visualize', command=self.run_pipeline).pack(pady=10)
        self.prog = ttk.Progressbar(frm, length=300, mode='determinate')
        self.prog.pack(pady=5)

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[('Keras .h5','*.h5')])
        if not path: return
        self.model, self.input_size = load_segmentation_model(path)
        messagebox.showinfo('Model loaded', f'Input size = {self.input_size}')

    def upload_images(self):
        folder = filedialog.askdirectory()
        if not folder: return
        files = sorted(glob.glob(os.path.join(folder, '*.jpg')))
        if not files:
            messagebox.showerror('Error', 'No JPG images found')
            return
        self.paths  = files
        self.images = [cv2.imread(f) for f in files]
        messagebox.showinfo('Loaded', f'{len(files)} images')

    def run_pipeline(self):
        if self.model is None:
            messagebox.showwarning('Please load the model first')
            return
        if not self.images:
            messagebox.showwarning('Please upload images first')
            return
        try:
            ld = float(self.ld.get())
            es = float(self.es.get())
        except:
            messagebox.showerror('Error', 'Invalid parameters')
            return

        threading.Thread(target=self._process, args=(ld,es), daemon=True).start()

    def _process(self, layer_dist, gauss_sigma):
        n = len(self.images)
        self.prog['maximum'] = n
        masked = []
        for img in self.images:
            m = segment_mask(self.model, self.input_size, img, thresh=0.5)
            masked.append(cv2.bitwise_and(img, img, mask=m))
            self.master.after(0, self.prog.step, 1)

        idx_map = focus_stack(masked)
        self.master.after(0, lambda: visualize_open3d(
            idx_map, layer_dist,
            xy_scale=1.0, z_scale=1.0,
            gauss_sigma=gauss_sigma
        ))

if __name__ == '__main__':
    root = Tk()
    FocusStackApp(root)
    root.mainloop()


#working code

import os
import glob
import threading

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.ndimage import median_filter, distance_transform_edt, gaussian_filter
from tkinter import Tk, Label, Button, Entry, filedialog, messagebox, Frame
from tkinter import ttk

import open3d as o3d
from matplotlib import cm

# ---------------- Segmentation Model ---------------- #
def load_segmentation_model(path):
    model = load_model(path, compile=False)
    _, h, w, _ = model.input_shape
    return model, (h, w)

# --------------- Masking via Segmentation --------------- #
def segment_mask(model, input_size, image, thresh=0.5):
    h0, w0 = image.shape[:2]
    h, w   = input_size
    roi    = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    inp    = np.expand_dims(roi.astype(np.float32)/255.0, axis=0)
    pred   = model.predict(inp, verbose=0)[0]
    prob   = pred[...,0] if pred.ndim==3 else pred
    m      = (prob>thresh).astype(np.uint8)*255

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(m)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(clean, [c], -1, 255, cv2.FILLED)
    return cv2.resize(clean,(w0,w0) if False else (w0,h0),interpolation=cv2.INTER_NEAREST)

# ---------------- Focus Stacking ---------------- #
def focus_stack(images):
    arr = np.stack(images, axis=0)
    n, h, w, _ = arr.shape

    measure = np.zeros((h,w),dtype=np.float32)
    idxs    = -1*np.ones((h,w),dtype=np.int32)

    for i in range(n):
        gray = cv2.cvtColor(arr[i],cv2.COLOR_BGR2GRAY)
        lap  = cv2.Laplacian(gray,cv2.CV_64F)
        m    = lap>measure
        measure[m] = lap[m]
        idxs[m]    = i

    valid = idxs>=0
    dist,(iy,ix) = distance_transform_edt(~valid,return_indices=True)
    filled = idxs.copy()
    filled[~valid] = idxs[iy[~valid], ix[~valid]]

    idx_map = median_filter(filled, size=3)
    return idx_map

# -------- Open3D Viz: Flat Interiors + Tapered Rim ---------- #
def visualize_open3d(idx_map, union_mask,
                     layer_dist, xy_scale=1.0, z_scale=1.0, edge_sigma=2):
    # 1) raw depth
    depth = idx_map.astype(np.float32)*layer_dist
    # 2) remove micro-spikes
    depth = median_filter(depth, size=5)

    # 3) find true interior vs rim of the *union* mask
    kernel   = np.ones((5,5),np.uint8)
    interior = cv2.erode(union_mask.astype(np.uint8),kernel,iterations=1).astype(bool)
    rim      = union_mask & ~interior

    # 4) gentle Gaussian on entire depth
    smooth = gaussian_filter(depth, sigma=edge_sigma)

    # 5) merge: interior flat, rim tapered
    final = depth.copy()
    final[rim] = smooth[rim]

    # 6) build point cloud just from union_mask
    h,w = final.shape
    xs = np.arange(w)*xy_scale
    ys = np.arange(h)*xy_scale
    Xg,Yg = np.meshgrid(xs,ys)
    Zg = final*z_scale

    pts = np.vstack((Xg[union_mask],Yg[union_mask],Zg[union_mask])).T
    norm = (Zg - Zg[union_mask].min())/(Zg[union_mask].ptp()+1e-8)
    cols = cm.hsv(norm)[..., :3][union_mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    o3d.visualization.draw_geometries([pcd], window_name="3D Cloth Focus Stack")

# -------------------- GUI App -------------------- #
class FocusStackApp:
    def __init__(self,master):
        self.master = master
        master.title('3D Focus Stack (Open3D)')
        master.geometry('800x600')

        self.model = None
        self.input_size = None
        self.images = []
        self.paths  = []

        frm = Frame(master); frm.pack(padx=10,pady=10)
        Button(frm, text='Load .h5 Model', command=self.load_model).pack(pady=5)
        Button(frm, text='Upload Images',   command=self.upload_images).pack(pady=5)

        Label(frm, text='Layer Distance:').pack(pady=2)
        self.ld = Entry(frm); self.ld.insert(0,'1.0'); self.ld.pack(pady=2)
        Label(frm, text='Rim Smooth σ:').pack(pady=2)
        self.es = Entry(frm); self.es.insert(0,'2');   self.es.pack(pady=2)

        Button(frm, text='Stack & Visualize', command=self.run).pack(pady=10)
        self.prog=ttk.Progressbar(frm,length=300,mode='determinate');self.prog.pack(pady=5)

    def load_model(self):
        p = filedialog.askopenfilename(filetypes=[('Keras .h5','*.h5')])
        if not p: return
        self.model,self.input_size = load_segmentation_model(p)
        messagebox.showinfo('Model','Loaded input size '+str(self.input_size))

    def upload_images(self):
        d = filedialog.askdirectory()
        if not d: return
        files = sorted(glob.glob(os.path.join(d,'*.jpg')))
        if not files:
            messagebox.showerror('Error','No JPGs found'); return
        self.paths=files
        self.images=[cv2.imread(f) for f in files]
        messagebox.showinfo('Images',f'{len(files)} loaded')
        # prepare union mask
        self.union_mask=None

    def run(self):
        if self.model is None:
            messagebox.showwarning('Load model first'); return
        if not self.images:
            messagebox.showwarning('Upload images first'); return
        try:
            ld = float(self.ld.get())
            es = float(self.es.get())
        except:
            messagebox.showerror('Error','Invalid params'); return
        threading.Thread(target=self._proc,args=(ld,es),daemon=True).start()

    def _proc(self,layer_dist,edge_sigma):
        n=len(self.images); self.prog['maximum']=n
        masked=[]
        # build union mask
        for img in self.images:
            m=segment_mask(self.model,self.input_size,img,thresh=0.5)>0
            if self.union_mask is None:
                self.union_mask = m.copy()
            else:
                self.union_mask |= m
            masked.append(cv2.bitwise_and(img,img,mask=m.astype(np.uint8)*255))
            self.master.after(0,self.prog.step,1)

        idx_map=focus_stack(masked)
        self.master.after(0,lambda: visualize_open3d(
            idx_map,self.union_mask,layer_dist,
            xy_scale=1.0,z_scale=1.0,edge_sigma=edge_sigma
        ))

if __name__=='__main__':
    root=Tk();FocusStackApp(root);root.mainloop()
