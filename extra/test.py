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
        # Use Otsu thresholding for robustness:
        _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary_mask

def mask_image_with_rcnn(image):
    """
    Use Mask R-CNN to obtain a segmentation mask.
    Falls back to traditional masking if RCNN fails.
    """
    try:
        mask = get_object_mask(image)
    except Exception as e:
        mask = traditional_masking(image)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image, mask

# ---------------------- Helper Functions ---------------------- #
def load_images_from_folder(folder_path):
    image_files = glob.glob(os.path.join(folder_path, '*.jpg'))
    if not image_files:
        raise FileNotFoundError(f"No images found in the folder: {folder_path}")
    image_files.sort()  # Ensure consistent ordering
    images = [cv2.imread(img_file) for img_file in image_files]
    return images

def traditional_masking(image):
    """
    Create a mask of in-focus regions using Sobel gradients on a CLAHE-normalized image.
    Uses Otsu's thresholding to adapt to lighting conditions.
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
        for y,x in zip(*np.where(new_pixels)):
            neighbors = filled[max(0, y-1):y+2, max(0, x-1):x+2]
            neighbors = neighbors[neighbors != -1]
            if len(neighbors) > 0:
                filled[y,x] = int(np.median(neighbors))
        valid_mask = (filled != -1).astype(np.uint8)
    return filled

def focus_stack(images):
    """
    Two-pass focus stack that includes a special second pass
    to re-check reflective areas (V>240) that might actually be in focus.
    """
    shape = images[0].shape[:2]
    focus_measure = np.full(shape, -np.inf, dtype=np.float32)
    focus_indices = np.full(shape, -1, dtype=np.int32)
    
    # ---------- 1ST PASS: Laplacian + Otsu + Reflect Mask ---------- #
    # We do as before to classify initial valid focus points.
    for i, image in enumerate(images):
        # 1) Segmentation (RCNN/traditional):
        _, seg_mask = mask_image_with_rcnn(image)
        seg_binary = (seg_mask > 0)

        # 2) Normalize lighting:
        norm_img = normalize_lighting(image)

        # 3) Laplacian measure:
        gray = cv2.cvtColor(norm_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3,3), 0)
        lap = cv2.Laplacian(blurred, cv2.CV_64F)
        measure = np.abs(lap)
        measure = np.nan_to_num(measure)

        # 4) Otsu thresholding on measure:
        measure_uint = cv2.normalize(measure, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, otsu_thresh = cv2.threshold(measure_uint, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        valid = measure_uint > otsu_thresh

        # 5) Reflective region check:
        hsv = cv2.cvtColor(norm_img, cv2.COLOR_BGR2HSV)
        reflect_mask = (hsv[:,:,2] > 240)
        # Combine with segmentation mask:
        valid = np.logical_and(seg_binary, np.logical_or(valid, reflect_mask))

        # 6) Update focus_measure:
        update = np.logical_and(valid, measure > focus_measure)
        focus_measure[update] = measure[update]
        focus_indices[update] = i

    # ---------- 2ND PASS: Check reflective pixels that remain excluded ---------- #
    # Some reflective pixels might not pass the 1st pass if they had "medium" measure
    # but truly are in focus. We can forcibly check them again with local variance or local gradient.
    for i, image in enumerate(images):
        # Build a mask of "reflective" pixels that are STILL -1 in focus_indices.
        norm_img = normalize_lighting(image)
        hsv = cv2.cvtColor(norm_img, cv2.COLOR_BGR2HSV)
        reflect_excluded = (hsv[:,:,2] > 240) & (focus_indices == -1)

        # For those reflect_excluded pixels, check if local variance or gradient is high enough:
        # We'll do a local variance measure on a 5x5 window:
        gray = cv2.cvtColor(norm_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # local_var = E(X^2) - [E(X)]^2 approach:
        # We'll box-filter gray and gray^2, then compute variance.
        kernel_size = 5
        avg = cv2.boxFilter(gray, ddepth=-1, ksize=(kernel_size,kernel_size))
        avg_sq = cv2.boxFilter(gray*gray, ddepth=-1, ksize=(kernel_size,kernel_size))
        var_map = avg_sq - (avg*avg)
        var_map = np.maximum(var_map, 0)
        
        # Set a threshold for local variance => forced in:
        # e.g. choose 30.0 (adjust as needed).
        # We'll just guess: if var_map[y, x] > local_var_thresh => in focus
        local_var_thresh = 30.0
        
        # Reflective but have var > local_var_thresh => forcibly set focus_indices => i
        y_idxs, x_idxs = np.where(reflect_excluded)
        for y, x in zip(y_idxs, x_idxs):
            if var_map[y, x] > local_var_thresh:
                focus_indices[y, x] = i

    # ---------- Fill holes + build stacked image ---------- #
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
    Find the largest connected component in the valid focus mask and fill holes
    within that region using inpainting. This creates a full volumetric structure.
    """
    valid = (focus_indices != -1).astype(np.uint8) * 255
    num_labels, labels = cv2.connectedComponents(valid)
    largest_label = None
    largest_area = 0
    for label in range(1, num_labels):
        area = np.sum(labels==label)
        if area > largest_area:
            largest_area = area
            largest_label = label
    if largest_label is None:
        return depth_map, focus_indices
    comp_mask = (labels == largest_label).astype(np.uint8)*255
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    missing = np.logical_and(comp_mask==255, focus_indices==-1).astype(np.uint8)*255
    inpainted = cv2.inpaint(depth_norm, missing, 3, cv2.INPAINT_TELEA)
    filled_depth = depth_norm.copy().astype(np.float32)
    filled_depth[comp_mask==255] = inpainted[comp_mask==255]
    new_focus = focus_indices.copy()
    valid_vals = focus_indices[focus_indices!=-1]
    if valid_vals.size > 0:
        median_val = int(np.median(valid_vals))
        new_focus[(comp_mask==255) & (focus_indices==-1)] = median_val
    filled_depth = cv2.normalize(filled_depth, None, depth_map.min(), depth_map.max(), cv2.NORM_MINMAX)
    return filled_depth, new_focus

def depth_map_to_point_cloud(depth_map, focus_indices, num_images, xy_scale=1.0, z_scale=1.0):
    """
    Convert the (filled) depth map into a 3D point cloud.
    Each valid pixel becomes a point with coordinates scaled by xy_scale and z_scale.
    A continuous "jet" colormap is applied based on height.
    """
    valid = focus_indices != -1
    ys, xs = np.nonzero(valid)
    zs = depth_map[ys, xs] * z_scale
    xs = xs * xy_scale
    ys = ys * xy_scale
    points = np.stack((xs, ys, zs), axis=-1)
    if zs.size == 0:
        colors = np.zeros((points.shape[0], 3))
    else:
        z_min, z_max = zs.min(), zs.max()
        z_range = z_max - z_min if (z_max - z_min) > 0 else 1.0
        norm_z = (zs - z_min) / z_range
        colormap = cm.get_cmap('jet')
        colors = colormap(norm_z)[:, :3]
    return points, colors

def calculate_dimensions(points):
    """
    Calculate and return the dimensions (length, breadth, height) of the object in the point cloud.
    """
    if points.size == 0:
        return 0,0,0
    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)
    return x_max - x_min, y_max - y_min, z_max - z_min

def process_focus_stack(images, threshold_factor=0.2, progress_callback=None, 
                        layer_distance=0.05, xy_scale=1.0, z_scale=1.0):
    """
    Process a focus stack from a list of images.
    
    Parameters:
        images: list of images (numpy arrays)
        threshold_factor: used for focus threshold (not active if using Otsu)
        progress_callback: function taking a single integer argument, called after processing.
        layer_distance: factor to convert image index to depth.
        xy_scale: scale factor for x and y coordinates.
        z_scale: scale factor for the depth (z coordinate).
        
    Returns a triple:
        (stacked_image, (point_cloud, colors), dimensions)
    """
    if progress_callback is None:
        progress_callback = lambda x: None
    stacked_img, focus_indices = focus_stack(images)
    progress_callback(len(images))
    depth_map = create_depth_map(focus_indices, layer_distance)
    filled_depth, new_focus = fill_largest_component(depth_map, focus_indices)
    pts, cols = depth_map_to_point_cloud(filled_depth, new_focus, num_images=len(images),
                                         xy_scale=xy_scale, z_scale=z_scale)
    dims = calculate_dimensions(pts)
    return stacked_img, (pts, cols), dims

# ---------------------- GUI Application ---------------------- #
class PointCloudApp:
    def __init__(self, master):
        self.master = master
        self.master.title('3D Point Cloud Reconstruction Tool')
        self.master.geometry('800x600')
        self.frame = Frame(master)
        self.frame.pack(padx=10, pady=10)
        
        Label(self.frame, text="3D Point Cloud Reconstruction", font=("Arial", 18, "bold")).pack(pady=10)
        Button(self.frame, text='Upload Images', command=self.upload_images).pack(pady=10)
        
        Label(self.frame, text="Focus Threshold Factor (e.g., 0.2):").pack(pady=5)
        self.threshold_entry = Entry(self.frame)
        self.threshold_entry.insert(0, "0.2")
        self.threshold_entry.pack(pady=5)
        
        self.process_button = Button(self.frame, text='Process Images', command=self.start_processing)
        self.process_button.pack(pady=10)
        self.dimension_label = Label(self.frame, text="")
        self.dimension_label.pack(pady=10)
        self.progress_bar = ttk.Progressbar(self.frame, length=300, mode='determinate')
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
                # Create mask folder with suffix "_mask"
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
        threading.Thread(target=self.run_point_cloud_processing, args=(threshold_factor,), daemon=True).start()
    
    def run_point_cloud_processing(self, threshold_factor):
        self.progress_bar["value"] = 0
        self.progress_bar["maximum"] = len(self.images)
        images_cleaned = []
        # Process each image to extract masked object; save mask to folder.
        for i, img in enumerate(self.images):
            masked_img, mask = mask_image_with_rcnn(img)
            images_cleaned.append(masked_img)
            mask_filename = os.path.join(self.mask_folder, f"mask_{i:03d}.png")
            cv2.imwrite(mask_filename, mask)
            self.master.after(0, self.progress_bar.step, 1)
        
        if not images_cleaned:
            self.master.after(0, self.dimension_label.config, {"text": "Error: No valid images after processing."})
            return
        
        stacked_img, (pts, cols), dims = process_focus_stack(images_cleaned, threshold_factor=threshold_factor,
                                                             layer_distance=0.05, xy_scale=0.01, z_scale=1.0)
        dimension_text = f'Length: {dims[0]:.2f} mm, Breadth: {dims[1]:.2f} mm, Height: {dims[2]:.2f} mm'
        self.master.after(0, self.dimension_label.config, {"text": dimension_text})
        gc.collect()
        self.visualize_point_cloud(pts, cols)

    def visualize_point_cloud(self, points, colors):
        if len(points) == 0:
            messagebox.showerror("Error", "No points to display in the point cloud.")
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])

# ---------------------- Main ---------------------- #
if __name__ == "__main__":
    root = Tk()
    app = PointCloudApp(root)
    root.mainloop()
