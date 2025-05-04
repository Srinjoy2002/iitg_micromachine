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
from scipy.interpolate import Rbf

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
    normalized = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return normalized

def preprocess_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.ToTensor()])
    return transform(image_rgb).unsqueeze(0).to(device)

def get_object_mask(image):
    with torch.no_grad():
        image_tensor = preprocess_image(image)
        prediction = model(image_tensor)[0]
        if 'masks' not in prediction or len(prediction['masks']) == 0:
            raise ValueError("No valid masks detected")
        threshold = 0.5
        if prediction['scores'][0] < threshold:
            raise ValueError("No masks with a score above threshold")
        mask = prediction['masks'][0, 0].mul(255).byte().cpu().numpy()
        _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary_mask

def traditional_masking(image):
    norm_img = normalize_lighting(image)
    gray = cv2.cvtColor(norm_img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    grad_mag = cv2.GaussianBlur(grad_mag, (5,5), 0)
    grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX)
    grad_norm = np.nan_to_num(grad_norm).astype(np.uint8)
    _, mask = cv2.threshold(grad_norm, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def mask_image_with_rcnn(image):
    try:
        mask = get_object_mask(image)
    except Exception:
        mask = traditional_masking(image)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image, mask

def load_images_from_folder(folder_path):
    image_files = glob.glob(os.path.join(folder_path, '*.jpg'))
    if not image_files:
        raise FileNotFoundError(f"No images found in folder: {folder_path}")
    image_files.sort()
    images = [cv2.imread(img_file) for img_file in image_files]
    return images

# ---------------------- Classical Focus Stacking ---------------------- #
def wavelet_focus_measure(gray, wavelet='db2'):
    coeffs2 = pywt.dwt2(gray, wavelet)
    cA, (cH, cV, cD) = coeffs2
    detail = cv2.magnitude(cH.astype(np.float32), cV.astype(np.float32))
    detail = cv2.normalize(detail, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sobelx = cv2.Sobel(detail, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(detail, cv2.CV_64F, 0, 1, ksize=3)
    measure = np.sqrt(sobelx**2 + sobely**2)
    if measure.shape != gray.shape:
        measure = cv2.resize(measure, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    return measure

def fill_focus_holes(focus_indices):
    mask_invalid = (focus_indices == -1)
    filled = focus_indices.copy()
    kernel = np.ones((3,3), np.uint8)
    valid_mask = (~mask_invalid).astype(np.uint8)
    for _ in range(5):
        dilated = cv2.dilate(valid_mask, kernel, iterations=1)
        diff = dilated - valid_mask
        new_pixels = np.logical_and(diff==1, mask_invalid)
        if not np.any(new_pixels):
            break
        for y, x in zip(*np.where(new_pixels)):
            neighbors = filled[max(0, y-1):y+2, max(0, x-1):x+2]
            neighbors = neighbors[neighbors != -1]
            if len(neighbors) > 0:
                filled[y, x] = int(np.median(neighbors))
        valid_mask = (filled != -1).astype(np.uint8)
    return filled

def classical_focus_stack(images):
    if not images:
        raise ValueError("No images provided.")
    h, w = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY).shape
    n = len(images)
    measure_map = np.zeros((n, h, w), dtype=np.float32)
    
    for i, img in enumerate(images):
        normed = normalize_lighting(img)
        gray = cv2.cvtColor(normed, cv2.COLOR_BGR2GRAY)
        measure_map[i] = wavelet_focus_measure(gray)
    
    focus_indices = np.full((h, w), -1, dtype=np.int32)
    best_measure = np.full((h, w), -np.inf, dtype=np.float32)
    
    for i in range(n):
        update = measure_map[i] > best_measure
        best_measure[update] = measure_map[i][update]
        focus_indices[update] = i
    
    # Invert the focus indices
    focus_indices = (n - 1) - focus_indices
    
    focus_indices_filled = fill_focus_holes(focus_indices)
    images_stack = np.stack(images, axis=0)
    Y, X = np.indices((h, w))
    stacked = np.zeros_like(images[0])
    valid = (focus_indices_filled != -1)
    stacked[valid] = images_stack[focus_indices_filled[valid], Y[valid], X[valid]]
    return stacked, focus_indices_filled
# ---------------------- Depth Processing ---------------------- #
def fill_depth_map_2d(depth_map, max_iterations=5):
    h, w = depth_map.shape
    depth_f = depth_map.copy()
    invalid_val = 0
    for _ in range(max_iterations):
        changed = False
        for y in range(h):
            for x in range(w):
                if depth_f[y, x] == invalid_val:
                    neighbors = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = y+dy, x+dx
                            if 0 <= ny < h and 0 <= nx < w:
                                val = depth_f[ny, nx]
                                if val != invalid_val:
                                    neighbors.append(val)
                    if len(neighbors) > 3:
                        depth_f[y, x] = float(np.mean(neighbors))
                        changed = True
        if not changed:
            break
    depth_f = cv2.normalize(depth_f, None, depth_map.min(), depth_map.max(), cv2.NORM_MINMAX)
    return depth_f.astype(np.float32)

def fill_depth_map_horizontal(depth_map):
    filled = depth_map.copy()
    h, w = depth_map.shape
    for i in range(h):
        row = filled[i, :]
        valid = row > 0
        if np.sum(valid) < 2:
            continue
        x = np.arange(w)
        filled[i, :] = np.interp(x, x[valid], row[valid])
    return filled

def create_depth_map(focus_indices, layer_distance):
    # Create depth map with inverted indices
    max_val = np.max(focus_indices[focus_indices != -1])
    inverted_indices = np.where(focus_indices == -1, -1, max_val - focus_indices)
    return np.where(inverted_indices == -1, 0, inverted_indices.astype(np.float32)*layer_distance)

# ---------------------- Visualization ---------------------- #
def visualize_depth_map_matplotlib(depth_map, xy_scale=1.0, z_scale=1.0, sigma=4):
    depth_smooth = gaussian_filter(depth_map, sigma=sigma)
    depth_smooth = median_filter(depth_smooth, size=3)
    
    h, w = depth_smooth.shape
    X, Y = np.meshgrid(np.arange(w)*xy_scale, np.arange(h)*xy_scale)
    
    # Invert Z-axis for display
    Z = (depth_smooth.max() - depth_smooth) * z_scale
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    norm = plt.Normalize(Z.min(), Z.max())
    colors = plt.cm.viridis(norm(Z))
    
    surf = ax.plot_surface(
        X, Y, Z,
        facecolors=colors,
        edgecolor='none',
        rstride=2,
        cstride=2,
        linewidth=0,
        alpha=0.9
    )
    
    ax.view_init(elev=35, azim=-45)
    ax.set_title("3D Surface Topography", fontsize=14)
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_zlabel("Z (µm)")
    
    fig.colorbar(surf, shrink=0.6, aspect=15)
    plt.tight_layout()
    plt.show()

    ax.view_init(elev=35, azim=-45)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_title("3D Surface Topography", fontsize=14, pad=20)
    ax.set_xlabel("X (µm)", fontsize=12, labelpad=12)
    ax.set_ylabel("Y (µm)", fontsize=12, labelpad=12)
    ax.set_zlabel("Z (µm)", fontsize=12, labelpad=12)
    
    cbar = fig.colorbar(surf, shrink=0.6, aspect=15, pad=0.08)
    cbar.set_label('Height (µm)', rotation=270, labelpad=20, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.show()

# ---------------------- Pipeline ---------------------- #
def process_stack_pipeline(images, layer_dist=0.05, xy_scale=0.01, z_scale=1.0):
    stacked_color, focus_idx = classical_focus_stack(images)
    depth_map = create_depth_map(focus_idx, layer_dist)
    vertical_filled = fill_depth_map_2d(depth_map, max_iterations=3)
    horizontal_filled = fill_depth_map_horizontal(vertical_filled)
    final_depth = fill_missing_depth_rbf(horizontal_filled)
    
    final_norm = cv2.normalize(final_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    filtered = cv2.bilateralFilter(final_norm, d=15, sigmaColor=100, sigmaSpace=100)
    final_depth = cv2.normalize(filtered, None, final_depth.min(), final_depth.max(), cv2.NORM_MINMAX).astype(np.float32)
    
    return final_depth

def fill_missing_depth_rbf(depth_map, downsample_factor=16):
    original_shape = depth_map.shape
    new_width = original_shape[1] // downsample_factor
    new_height = original_shape[0] // downsample_factor
    small = cv2.resize(depth_map, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    X, Y = np.meshgrid(np.arange(new_width), np.arange(new_height))
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    depth_flat = small.flatten()
    
    valid = depth_flat > 0
    if np.sum(valid) < 3:
        return depth_map
    
    rbf = Rbf(X_flat[valid], Y_flat[valid], depth_flat[valid], 
              function='gaussian', smooth=10)
    
    filled_small = rbf(X, Y)
    filled_small = cv2.normalize(filled_small, None, small.min(), small.max(), cv2.NORM_MINMAX)
    
    filled_full = cv2.resize(filled_small, (original_shape[1], original_shape[0]), 
                        interpolation=cv2.INTER_CUBIC)
    
    return filled_full.astype(np.float32)

# ---------------------- GUI ---------------------- #
class My3DApp:
    def __init__(self, master):
        self.master = master
        self.master.title("3D Reconstruction Tool")
        self.master.geometry("800x600")
        self.frame = Frame(master)
        self.frame.pack(padx=10, pady=10)

        Label(self.frame, text="3D Structure Reconstruction", font=("Arial", 18, "bold")).pack(pady=10)
        Button(self.frame, text="Upload Images", command=self.upload).pack(pady=10)

        self.layer_entry = Entry(self.frame)
        self.layer_entry.insert(0, "0.05")
        self.layer_entry.pack(pady=5)
        self.xy_entry = Entry(self.frame)
        self.xy_entry.insert(0, "0.01")
        self.xy_entry.pack(pady=5)
        self.z_entry = Entry(self.frame)
        self.z_entry.insert(0, "1.0")
        self.z_entry.pack(pady=5)

        self.process_btn = Button(self.frame, text="Process Images", command=self.start_processing)
        self.process_btn.pack(pady=10)
        self.dim_label = Label(self.frame, text="")
        self.dim_label.pack(pady=10)
        self.prog_bar = ttk.Progressbar(self.frame, length=300, mode="determinate")
        self.prog_bar.pack(pady=10)
        self.images = []

    def upload(self):
        folder_path = filedialog.askdirectory(title="Select Folder with Images")
        if folder_path:
            try:
                self.images = load_images_from_folder(folder_path)
                messagebox.showinfo("Info", f"Loaded {len(self.images)} images.")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def start_processing(self):
        if not self.images:
            messagebox.showwarning("Warning", "Please upload images first.")
            return
        try:
            layer_val = float(self.layer_entry.get())
            xy_val = float(self.xy_entry.get())
            z_val = float(self.z_entry.get())
        except:
            messagebox.showerror("Error", "Invalid numeric input.")
            return
        threading.Thread(target=self.run_pipeline, args=(layer_val, xy_val, z_val), daemon=True).start()

    def run_pipeline(self, layer_dist, xy_scale, z_scale):
        self.prog_bar["value"] = 0
        self.prog_bar["maximum"] = len(self.images)
        try:
            final_depth = process_stack_pipeline(self.images, layer_dist, xy_scale, z_scale)
            self.master.after(0, lambda: [
                self.dim_label.config(text="Processing Complete"),
                visualize_depth_map_matplotlib(final_depth, xy_scale, z_scale)
            ])
        except Exception as e:
            self.master.after(0, messagebox.showerror, 'Error', str(e))

def main():
    root = Tk()
    app = My3DApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
