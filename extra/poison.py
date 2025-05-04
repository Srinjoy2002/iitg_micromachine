import sys
import cv2
import numpy as np
import glob
import threading
from tkinter import Tk, Label, Button, Entry, filedialog, messagebox, Frame
from tkinter import ttk
import open3d as o3d
import os
import gc
import matplotlib.cm as cm

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
    """
    Normalize lighting using CLAHE on the L-channel in LAB.
    This reduces overexposure due to LED illumination.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    normalized = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return normalized

def preprocess_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.ToTensor()])
    return transform(image_rgb).unsqueeze(0).to(device)

def get_object_mask(image):
    """
    Uses Mask R-CNN to get a segmentation mask for the object.
    Returns a binary mask using Otsu thresholding.
    """
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

def mask_image_with_rcnn(image):
    """
    Use Mask R-CNN to obtain a segmentation mask.
    Falls back to traditional masking if RCNN fails.
    """
    try:
        mask = get_object_mask(image)
    except Exception:
        mask = traditional_masking(image)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image, mask

# ---------------------- Helper Functions ---------------------- #
def load_images_from_folder(folder_path):
    image_files = glob.glob(os.path.join(folder_path, '*.jpg'))
    if not image_files:
        raise FileNotFoundError(f"No images found in the folder: {folder_path}")
    image_files.sort()
    images = [cv2.imread(img_file) for img_file in image_files]
    return images

def traditional_masking(image):
    """
    Create a mask of in-focus regions using Sobel gradients on a CLAHE-normalized image.
    Uses Otsu's thresholding.
    """
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

def fill_focus_holes(focus_indices):
    """
    Fill holes in the focus index map by using the median of valid neighbors.
    """
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

def focus_stack(images):
    """
    Two-pass focus stack using Laplacian+Otsu and reflective re-check.
    """
    shape = images[0].shape[:2]
    focus_measure = np.full(shape, -np.inf, dtype=np.float32)
    focus_indices = np.full(shape, -1, dtype=np.int32)
    
    # First pass
    for i, image in enumerate(images):
        _, seg_mask = mask_image_with_rcnn(image)
        seg_binary = (seg_mask > 0)
        norm_img = normalize_lighting(image)
        gray = cv2.cvtColor(norm_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3,3), 0)
        lap = cv2.Laplacian(blurred, cv2.CV_64F)
        measure = np.abs(lap)
        measure = np.nan_to_num(measure)
        measure_uint = cv2.normalize(measure, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, otsu_thresh = cv2.threshold(measure_uint, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        valid = measure_uint > otsu_thresh
        hsv = cv2.cvtColor(norm_img, cv2.COLOR_BGR2HSV)
        reflect_mask = (hsv[:,:,2] > 240)
        valid = np.logical_and(seg_binary, np.logical_or(valid, reflect_mask))
        update = np.logical_and(valid, measure > focus_measure)
        focus_measure[update] = measure[update]
        focus_indices[update] = i

    # Second pass: re-check reflective pixels
    for i, image in enumerate(images):
        norm_img = normalize_lighting(image)
        hsv = cv2.cvtColor(norm_img, cv2.COLOR_BGR2HSV)
        reflect_excluded = (hsv[:,:,2] > 240) & (focus_indices == -1)
        gray = cv2.cvtColor(norm_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        kernel_size = 5
        avg = cv2.boxFilter(gray, ddepth=-1, ksize=(kernel_size, kernel_size))
        avg_sq = cv2.boxFilter(gray*gray, ddepth=-1, ksize=(kernel_size, kernel_size))
        var_map = np.maximum(avg_sq - (avg*avg), 0)
        local_var_thresh = 30.0
        y_idxs, x_idxs = np.where(reflect_excluded)
        for y, x in zip(y_idxs, x_idxs):
            if var_map[y, x] > local_var_thresh:
                focus_indices[y, x] = i

    focus_indices_filled = fill_focus_holes(focus_indices)
    images_stack = np.stack(images, axis=0)
    Y, X = np.indices(shape)
    stacked = np.zeros_like(images[0])
    valid_pixels = (focus_indices_filled != -1)
    stacked[valid_pixels] = images_stack[focus_indices_filled[valid_pixels], Y[valid_pixels], X[valid_pixels]]
    return stacked, focus_indices_filled

def create_depth_map(focus_indices, layer_distance):
    """
    Create a depth map by multiplying the focus index by layer_distance.
    """
    depth_map = np.where(focus_indices == -1, 0, focus_indices.astype(np.float32) * layer_distance)
    return depth_map

def fill_largest_component(depth_map, focus_indices):
    """
    Inpaint holes in the largest connected component of the focus mask.
    """
    valid = (focus_indices != -1).astype(np.uint8) * 255
    num_labels, labels = cv2.connectedComponents(valid)
    largest_label = None
    largest_area = 0
    for label in range(1, num_labels):
        area = np.sum(labels == label)
        if area > largest_area:
            largest_area = area
            largest_label = label
    if largest_label is None:
        return depth_map, focus_indices
    comp_mask = (labels == largest_label).astype(np.uint8) * 255
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    missing = np.logical_and(comp_mask == 255, focus_indices == -1).astype(np.uint8) * 255
    inpainted = cv2.inpaint(depth_norm, missing, 3, cv2.INPAINT_TELEA)
    filled_depth = depth_norm.copy().astype(np.float32)
    filled_depth[comp_mask == 255] = inpainted[comp_mask == 255]
    new_focus = focus_indices.copy()
    valid_vals = focus_indices[focus_indices != -1]
    if valid_vals.size > 0:
        median_val = int(np.median(valid_vals))
        new_focus[(comp_mask == 255) & (focus_indices == -1)] = median_val
    filled_depth = cv2.normalize(filled_depth, None, depth_map.min(), depth_map.max(), cv2.NORM_MINMAX)
    return filled_depth, new_focus

def morphological_close_depth(depth_map, kernel_size=9, iterations=2):
    """
    Apply multiple morphological closings to the depth map with a large kernel.
    """
    depth_8u = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = depth_8u.copy()
    for _ in range(iterations):
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
    closed = cv2.normalize(closed, None, depth_map.min(), depth_map.max(), cv2.NORM_MINMAX)
    return closed.astype(np.float32)

def fill_depth_iterative(depth_map, max_iterations=3):
    """
    Iteratively fill zero-valued pixels by averaging neighbors.
    A lower maximum iterations value is used to reduce over-smoothing.
    """
    h, w = depth_map.shape
    depth_f = depth_map.copy()
    for _ in range(max_iterations):
        changed = False
        for y in range(h):
            for x in range(w):
                if depth_f[y, x] == 0:
                    neighbors = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny = y + dy
                            nx = x + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                val = depth_f[ny, nx]
                                if val != 0:
                                    neighbors.append(val)
                    if len(neighbors) >= 3:
                        depth_f[y, x] = float(np.mean(neighbors))
                        changed = True
        if not changed:
            break
    depth_f = cv2.normalize(depth_f, None, depth_map.min(), depth_map.max(), cv2.NORM_MINMAX)
    return depth_f.astype(np.float32)

def depth_map_to_point_cloud(depth_map, focus_indices, num_images, xy_scale=1.0, z_scale=1.0):
    """
    Converts the (filled) depth map into a 3D point cloud.
    """
    valid = focus_indices != -1
    ys, xs = np.nonzero(valid)
    zs = depth_map[ys, xs] * z_scale
    xs = xs * xy_scale
    ys = ys * xy_scale
    points = np.stack((xs, ys, zs), axis=-1)
    return points

def color_by_height(points):
    """
    Assign a continuous "jet" colormap based on point Z-heights.
    """
    if points.size == 0:
        return np.zeros((0, 3))
    z_vals = points[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    z_range = z_max - z_min if z_max > z_min else 1.0
    norm_z = (z_vals - z_min) / z_range
    colormap = cm.get_cmap('jet')
    colors = colormap(norm_z)[:, :3]
    return colors

def calculate_dimensions(points):
    """
    Calculate (length, breadth, height) from the 3D points.
    """
    if points.size == 0:
        return 0, 0, 0
    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)
    return x_max - x_min, y_max - y_min, z_max - z_min

def create_alpha_shape_mesh(points, alpha=0.02):
    """
    Create a 3D structure (mesh) from a point cloud using Alpha Shapes.
    Adjust the alpha parameter to balance detail versus connectivity.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        mesh.compute_vertex_normals()
        return mesh
    except Exception as e:
        print("Alpha shape reconstruction failed:", e)
        return None

def process_focus_stack(images, threshold_factor=0.2, progress_callback=None,
                        layer_distance=0.05, xy_scale=1.0, z_scale=1.0):
    """
    Process a focus stack and produce a 3D structure as an alpha shape mesh.
    Returns: (stacked_image, mesh, dimensions)
    """
    if progress_callback is None:
        progress_callback = lambda x: None

    # 1) Compute focus stack
    stacked_img, focus_indices = focus_stack(images)
    progress_callback(len(images))

    # 2) Create depth map and fill largest component
    depth_map = create_depth_map(focus_indices, layer_distance)
    filled_depth, new_focus = fill_largest_component(depth_map, focus_indices)
    
    # 3) Morphological close
    closed_depth = morphological_close_depth(filled_depth, kernel_size=9, iterations=2)
    
    # 4) Iterative fill to reduce gaps (less aggressive to preserve curvature)
    final_depth = fill_depth_iterative(closed_depth, max_iterations=3)

    # 5) Convert to 3D point cloud
    pts = depth_map_to_point_cloud(final_depth, new_focus, num_images=len(images),
                                   xy_scale=xy_scale, z_scale=z_scale)
    
    # 6) Generate 3D structure using alpha shape (captures curvature)
    mesh = create_alpha_shape_mesh(pts, alpha=0.02)
    
    dims = calculate_dimensions(pts)
    return stacked_img, mesh, dims

# ---------------------- GUI Application ---------------------- #
class PointCloudApp:
    def __init__(self, master):
        self.master = master
        self.master.title("3D Structure Reconstruction Tool")
        self.master.geometry("800x600")
        self.frame = Frame(master)
        self.frame.pack(padx=10, pady=10)

        Label(self.frame, text="3D Structure Reconstruction", font=("Arial", 18, "bold")).pack(pady=10)
        Button(self.frame, text="Upload Images", command=self.upload_images).pack(pady=10)

        Label(self.frame, text="Focus Threshold Factor (e.g., 0.2):").pack(pady=5)
        self.threshold_entry = Entry(self.frame)
        self.threshold_entry.insert(0, "0.2")
        self.threshold_entry.pack(pady=5)

        self.process_button = Button(self.frame, text="Process Images", command=self.start_processing)
        self.process_button.pack(pady=10)
        self.dimension_label = Label(self.frame, text="")
        self.dimension_label.pack(pady=10)
        self.progress_bar = ttk.Progressbar(self.frame, length=300, mode="determinate")
        self.progress_bar.pack(pady=10)
        self.images = []
        self.folder_path = None
        self.mask_folder = None

    def upload_images(self):
        folder_path = filedialog.askdirectory(title="Select Folder with Images")
        if folder_path:
            try:
                self.folder_path = folder_path
                self.images = load_images_from_folder(folder_path)
                self.mask_folder = folder_path + "_mask"
                if not os.path.exists(self.mask_folder):
                    os.makedirs(self.mask_folder)
                messagebox.showinfo("Info", f"{len(self.images)} images loaded.")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def start_processing(self):
        if not self.images:
            messagebox.showwarning("Warning", "Please upload images first.")
            return
        try:
            threshold_factor = float(self.threshold_entry.get())
        except:
            messagebox.showerror("Error", "Invalid threshold factor. Please enter a valid number.")
            return
        threading.Thread(target=self.run_structure_processing, args=(threshold_factor,), daemon=True).start()

    def run_structure_processing(self, threshold_factor):
        self.progress_bar["value"] = 0
        self.progress_bar["maximum"] = len(self.images)
        images_cleaned = []
        for i, img in enumerate(self.images):
            masked_img, mask = mask_image_with_rcnn(img)
            images_cleaned.append(masked_img)
            if self.mask_folder:
                mask_filename = os.path.join(self.mask_folder, f"mask_{i:03d}.png")
                cv2.imwrite(mask_filename, mask)
            self.master.after(0, self.progress_bar.step, 1)
        
        if not images_cleaned:
            self.master.after(0, self.dimension_label.config, {"text": "Error: No valid images after processing."})
            return

        # Full pipeline: compute 3D structure via alpha shape reconstruction.
        stacked_img, mesh, dims = process_focus_stack(images_cleaned,
                                                       threshold_factor=threshold_factor,
                                                       layer_distance=0.05,
                                                       xy_scale=0.01,
                                                       z_scale=1.0)
        dimension_text = f"Length: {dims[0]:.2f} mm, Breadth: {dims[1]:.2f} mm, Height: {dims[2]:.2f} mm"
        self.master.after(0, self.dimension_label.config, {"text": dimension_text})
        gc.collect()
        self.visualize_mesh(mesh)

    def visualize_mesh(self, mesh):
        if mesh is None or len(mesh.vertices) == 0:
            messagebox.showerror("Error", "No 3D structure to display.")
            return
        o3d.visualization.draw_geometries([mesh])

# ---------------------- Main ---------------------- #
if __name__ == "__main__":
    root = Tk()
    app = PointCloudApp(root)
    root.mainloop()
