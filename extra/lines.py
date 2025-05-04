import sys
import cv2
import numpy as np
import glob
import threading
from tkinter import Tk, Label, Button, Entry, filedialog, messagebox, Frame
from tkinter import ttk
import os
import gc
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from skimage.transform import resize
import pywt
from scipy.ndimage import gaussian_filter, median_filter
from scipy.interpolate import Rbf, griddata

# ---------------------- Model Initialization ---------------------- #
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision import transforms as T

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()
model.to(device)

# ---------------------- Preprocessing Functions ---------------------- #
def normalize_lighting(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def preprocess_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    t = T.Compose([T.ToTensor()])
    return t(image_rgb).unsqueeze(0).to(device)

def get_object_mask(image):
    with torch.no_grad():
        inp = preprocess_image(image)
        pred = model(inp)[0]
        if not pred['masks'].size(0) or pred['scores'][0] < 0.5:
            raise ValueError
        m = pred['masks'][0,0].mul(255).byte().cpu().numpy()
        _, bm = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return bm

def traditional_masking(image):
    norm = normalize_lighting(image)
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, 5)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, 5)
    mag = np.sqrt(sx**2 + sy**2)
    blur = cv2.GaussianBlur(mag, (5,5), 0)
    norm2 = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m = cv2.threshold(norm2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

def mask_image_with_rcnn(image):
    try:
        mask = get_object_mask(image)
    except:
        mask = traditional_masking(image)
    return cv2.bitwise_and(image, image, mask=mask), mask

def load_images_from_folder(folder):
    files = sorted(glob.glob(os.path.join(folder, '*.jpg')))
    if not files:
        raise FileNotFoundError(f"No images in {folder}")
    return [cv2.imread(f) for f in files]

# ---------------------- Focus Stacking ---------------------- #
def wavelet_focus_measure(gray):
    cA,(cH,cV,cD) = pywt.dwt2(gray, 'db2')
    detail = cv2.magnitude(cH.astype(np.float32), cV.astype(np.float32))
    detail = cv2.normalize(detail, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sx = cv2.Sobel(detail, cv2.CV_64F, 1, 0, 3)
    sy = cv2.Sobel(detail, cv2.CV_64F, 0, 1, 3)
    m = np.sqrt(sx**2 + sy**2)
    return cv2.resize(m, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR) \
           if m.shape != gray.shape else m

def fill_focus_holes(fi):
    inv = (fi == -1)
    out = fi.copy()
    for _ in range(5):
        vm = (~inv).astype(np.uint8)
        dl = cv2.dilate(vm, np.ones((3,3),np.uint8))
        newpix = (dl - vm) == 1
        if not newpix.any(): break
        for y,x in zip(*np.where(newpix)):
            nbr = out[max(0,y-1):y+2, max(0,x-1):x+2]
            vals = nbr[nbr != -1]
            if vals.size:
                out[y,x] = int(np.median(vals))
        inv = (out == -1)
    return out

def classical_focus_stack(imgs):
    h, w = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY).shape
    n = len(imgs)
    fm = np.zeros((n,h,w), np.float32)
    for i,img in enumerate(imgs):
        gray = cv2.cvtColor(normalize_lighting(img), cv2.COLOR_BGR2GRAY)
        fm[i] = wavelet_focus_measure(gray)

    best = np.full((h,w), -np.inf)
    idx  = np.full((h,w), -1, dtype=np.int32)
    for i in range(n):
        up = fm[i] > best
        best[up] = fm[i][up]
        idx[up]  = i

    idx = (n - 1) - idx
    idx_f = fill_focus_holes(idx)

    stack = np.zeros_like(imgs[0])
    Y,X = np.indices((h,w))
    mask = (idx_f != -1)
    stack[mask] = np.stack(imgs)[idx_f[mask], Y[mask], X[mask]]
    return stack, idx_f

# ---------------------- Depth Map ---------------------- #
def create_depth_map(fi, layer_dist):
    mx = np.max(fi[fi != -1])
    inv = np.where(fi == -1, -1, mx - fi)
    return np.where(inv == -1, 0, inv.astype(np.float32) * layer_dist)

def fill_largest_comp_inpaint(dm, fi):
    valid = (fi != -1).astype(np.uint8)*255
    n_lbl, lbls = cv2.connectedComponents(valid)
    areas = [(np.sum(lbls==l), l) for l in range(1,n_lbl)]
    if not areas: return dm, fi
    _, lab = max(areas)
    comp = (lbls == lab).astype(np.uint8)*255
    comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    dn = cv2.normalize(dm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    miss = np.logical_and(comp==255, fi==-1).astype(np.uint8)*255
    inp = cv2.inpaint(dn, miss, 5, cv2.INPAINT_TELEA)
    out = dn.astype(np.float32)
    out[comp==255] = inp[comp==255]
    fi2 = fi.copy()
    med = int(np.median(fi[fi!=-1]))
    fi2[np.logical_and(comp==255, fi==-1)] = med
    return cv2.normalize(out, None, dm.min(), dm.max(), cv2.NORM_MINMAX), fi2

def fill_depth_map_2d(dm):
    h,w=dm.shape; out=dm.copy()
    for _ in range(3):
        ch=False
        for y in range(h):
            for x in range(w):
                if out[y,x]==0:
                    nbr = out[max(0,y-1):y+2, max(0,x-1):x+2]
                    vals = nbr[nbr>0]
                    if vals.size>3:
                        out[y,x] = float(vals.mean()); ch=True
        if not ch: break
    return cv2.normalize(out, None, dm.min(), dm.max(), cv2.NORM_MINMAX)

def fill_missing_depth_rbf(dm):
    h,w=dm.shape
    small = cv2.resize(dm, (w//4, h//4), interpolation=cv2.INTER_LINEAR)
    X,Y = np.meshgrid(np.arange(w//4), np.arange(h//4))
    pts = np.stack((X.flatten(), Y.flatten()),1)
    vals = small.flatten(); mask = vals>0
    if mask.sum()<3: return dm
    rbf = Rbf(pts[mask,0], pts[mask,1], vals[mask], function='gaussian', smooth=10)
    Z = rbf(X,Y)
    Zn = cv2.normalize(Z, None, small.min(), small.max(), cv2.NORM_MINMAX)
    return cv2.resize(Zn, (w,h), interpolation=cv2.INTER_CUBIC).astype(np.float32)

# ---------------------- Visualization ---------------------- #
def visualize_depth_map_matplotlib(depth_map, stacked_color=None,
                                   xy_scale=1.0, z_scale=1.0,
                                   top_roughness_frac=0.1,
                                   top_frac=None):
    # alias support
    if top_frac is not None:
        top_roughness_frac = top_frac

    # up‑sample + smooth
    h, w = depth_map.shape
    uf = 4
    new_h, new_w = h*uf, w*uf
    xs = np.linspace(0, (w-1)*xy_scale, new_w)
    ys = np.linspace(0, (h-1)*xy_scale, new_h)
    Xq, Yq = np.meshgrid(xs, ys)

    X, Y = np.meshgrid(np.arange(w)*xy_scale, np.arange(h)*xy_scale)
    pts = np.stack([X.ravel(), Y.ravel()]).T
    Zq = griddata(pts, depth_map.ravel(), (Xq, Yq), method='cubic', fill_value=0)
    Zq = gaussian_filter(Zq, sigma=2.0)

    zn, zx = Zq.min(), Zq.max()
    C = cm.jet((Zq - zn)/(zx - zn + 1e-6))[..., :3]

    if stacked_color is not None:
        gray = cv2.cvtColor(stacked_color, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 200)
        edges_up = cv2.resize(edges, (new_w, new_h), interpolation=cv2.INTER_NEAREST)>0
        C[edges_up] = [0,0,0]

    fig = plt.figure(figsize=(9,7))
    ax  = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(
        Xq, Yq, Zq * z_scale,
        facecolors=C,
        rcount=new_h, ccount=new_w,
        linewidth=0, antialiased=False, shade=True,
        edgecolor='none'
    )

    ax.set_box_aspect((1,1,0.5))
    ax.view_init(elev=35, azim=-45)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.grid(False)
    plt.tight_layout()
    plt.show()

# ---------------------- Pipeline & GUI ---------------------- #
def process_stack_pipeline(images, layer_dist, xy_scale, z_scale):
    sc, fi = classical_focus_stack(images)
    dm = create_depth_map(fi, layer_dist)
    dm2, fi2 = fill_largest_comp_inpaint(dm, fi)
    dm3 = fill_depth_map_2d(dm2)
    dm4 = fill_missing_depth_rbf(dm3)
    return sc, dm4

class My3DApp:
    def __init__(self, master):
        self.master = master
        master.title("3D Cloth‑like Reconstruction")
        self.frame = Frame(master)
        self.frame.pack(padx=10, pady=10)
        Button(self.frame, text="Upload Images", command=self.upload).pack(pady=5)
        Label(self.frame, text="Layer Distance (mm):").pack()
        self.le = Entry(self.frame); self.le.insert(0, "0.05"); self.le.pack(pady=2)
        Label(self.frame, text="Top Roughness Fraction:").pack()
        self.re = Entry(self.frame); self.re.insert(0, "0.1"); self.re.pack(pady=2)
        Button(self.frame, text="Process & Visualize", command=self.start).pack(pady=10)
        self.images = []

    def upload(self):
        fld = filedialog.askdirectory(title="Select Folder")
        if fld:
            try:
                self.images = load_images_from_folder(fld)
                messagebox.showinfo("Loaded", f"{len(self.images)} images")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def start(self):
        if not self.images:
            return messagebox.showwarning("Warning", "Upload first")
        ld = float(self.le.get()); rf = float(self.re.get())
        sc, dm = process_stack_pipeline(self.images, ld, 0.01, 1.0)
        visualize_depth_map_matplotlib(dm, sc,
                                       xy_scale=0.01, z_scale=1.0,
                                       top_frac=rf)

def main():
    root = Tk()
    My3DApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
