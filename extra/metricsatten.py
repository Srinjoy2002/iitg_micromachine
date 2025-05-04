import sys
import cv2
import numpy as np
import glob
import threading
from tkinter import Tk, Label, Button, filedialog, messagebox, Frame
from tkinter import ttk
import open3d as o3d
import os
import gc
import matplotlib.cm as cm

# ---------------------- Helper Functions ---------------------- #
def load_images_from_folder(folder_path):
    image_files = glob.glob(os.path.join(folder_path, '*.jpg'))
    if not image_files:
        raise FileNotFoundError(f"No images found in the folder: {folder_path}")
    # Sorting ensures consistent ordering
    image_files.sort()
    images = [cv2.imread(img_file) for img_file in image_files]
    return images

def traditional_masking(image):
    """
    Create a mask of in-focus regions using Sobel gradients.
    (This is only used as a rough mask to exclude background.)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    grad_mag = cv2.GaussianBlur(grad_mag, (5, 5), 0)
    grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX)
    grad_norm = np.nan_to_num(grad_norm).astype(np.uint8)
    _, mask = cv2.threshold(grad_norm, 0.3 * grad_norm.max(), 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def fill_focus_holes(focus_indices):
    """
    Fill holes in the focus index map by using the median of valid neighbors.
    """
    mask_invalid = (focus_indices == -1)
    focus_indices_filled = focus_indices.copy()
    kernel = np.ones((3, 3), np.uint8)
    valid_mask = (~mask_invalid).astype(np.uint8)
    for _ in range(5):
        dilated_valid = cv2.dilate(valid_mask, kernel, iterations=1)
        diff = dilated_valid - valid_mask
        new_pixels = np.logical_and(diff == 1, mask_invalid)
        if not np.any(new_pixels):
            break
        for y, x in zip(*np.where(new_pixels)):
            neighbors = focus_indices_filled[max(0, y-1):y+2, max(0, x-1):x+2]
            neighbors = neighbors[neighbors != -1]
            if len(neighbors) > 0:
                focus_indices_filled[y, x] = int(np.median(neighbors))
        valid_mask = (focus_indices_filled != -1).astype(np.uint8)
    return focus_indices_filled

def focus_stack(images):
    """
    Compute a focus stack using the Laplacian variance as the focus measure.
    Pixels from the image with the highest focus (sharpness) value at that position are chosen.
    Returns the stacked image and focus index map.
    """
    stack_shape = images[0].shape[:2]
    focus_measure = np.full(stack_shape, -np.inf, dtype=np.float32)
    focus_indices = np.full(stack_shape, -1, dtype=int)

    for i, image in enumerate(images):
        # Use a rough mask to limit computations (optional)
        mask = traditional_masking(image)
        mask_bool = mask > 0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        # Laplacian as focus measure
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        focus_score = np.absolute(laplacian)
        focus_score = np.nan_to_num(focus_score)

        # Threshold per image: consider pixels above 30% of this image's maximum focus score
        current_max = np.max(focus_score)
        if current_max == 0:
            continue
        thresh = 0.25 * current_max
        valid = focus_score > thresh
        update_mask = np.logical_and(valid, focus_score > focus_measure)
        focus_measure[update_mask] = focus_score[update_mask]
        focus_indices[update_mask] = i

    focus_indices_filled = fill_focus_holes(focus_indices)

    # Build the stacked image using the best-focus pixel from each image.
    images_stack = np.stack(images, axis=0)
    Y, X = np.indices(stack_shape)
    stacked_image = np.zeros_like(images[0])
    valid_pixels = (focus_indices_filled != -1)
    stacked_image[valid_pixels] = images_stack[focus_indices_filled[valid_pixels], Y[valid_pixels], X[valid_pixels]]
    
    return stacked_image, focus_indices_filled

def create_depth_map(focus_indices, layer_distance):
    """
    Create a depth map from the focus index map.
    Each pixel’s depth is the image index multiplied by layer_distance.
    """
    depth_map = np.where(focus_indices == -1, 0, focus_indices.astype(np.float32) * layer_distance)
    return depth_map

def depth_map_to_point_cloud(depth_map, focus_indices, num_images, xy_scale=1.0, z_scale=1.0):
    """
    Converts the depth map to a 3D point cloud.
    Each valid pixel becomes a 3D point with coordinates scaled by xy_scale and z_scale.
    A continuous height-mapped colormap is applied (using Matplotlib's "jet" colormap).
    """
    valid_mask = focus_indices != -1
    ys, xs = np.nonzero(valid_mask)
    zs = depth_map[ys, xs] * z_scale
    xs = xs * xy_scale
    ys = ys * xy_scale
    points = np.stack((xs, ys, zs), axis=-1)

    if zs.size == 0:
        colors = np.zeros((points.shape[0], 3))
    else:
        # Normalize the height (z) values to [0, 1]
        z_min = zs.min()
        z_max = zs.max()
        z_range = z_max - z_min if (z_max - z_min) > 0 else 1.0
        norm_z = (zs - z_min) / z_range

        # Use Matplotlib's "jet" colormap (returns RGBA; take only RGB components)
        colormap = cm.get_cmap('jet')
        colors = colormap(norm_z)[:, :3]  # shape: (n_points, 3)
    
    return points, colors

def calculate_dimensions(points):
    """
    Calculate and return the dimensions (length, breadth, height) of the object in the point cloud.
    """
    if points.size == 0:
        return 0, 0, 0
    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)
    return x_max - x_min, y_max - y_min, z_max - z_min

# ---------------------- GUI Application ---------------------- #
class PointCloudApp:
    def __init__(self, master):
        self.master = master
        self.master.title("3D Point Cloud Reconstruction Tool")
        self.master.geometry("800x600")
        self.frame = Frame(master)
        self.frame.pack(padx=10, pady=10)

        Label(self.frame, text="3D Point Cloud Reconstruction", font=("Arial", 18, "bold")).pack(pady=10)
        Button(self.frame, text="Upload Images", command=self.upload_images).pack(pady=10)
        self.process_button = Button(self.frame, text="Process Images", command=self.start_processing)
        self.process_button.pack(pady=10)
        self.dimension_label = Label(self.frame, text="")
        self.dimension_label.pack(pady=10)
        self.progress_bar = ttk.Progressbar(self.frame, length=300, mode="determinate")
        self.progress_bar.pack(pady=10)
        self.images = []

    def upload_images(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            try:
                self.images = load_images_from_folder(folder_path)
                messagebox.showinfo("Info", f"{len(self.images)} images loaded.")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def start_processing(self):
        if not self.images:
            messagebox.showwarning("Warning", "Please upload images first.")
            return
        threading.Thread(target=self.run_point_cloud_processing, daemon=True).start()

    def run_point_cloud_processing(self):
        # Reset progress bar values on the main thread.
        self.master.after(0, self.progress_bar.configure, {"value": 0})
        self.master.after(0, self.progress_bar.configure, {"maximum": len(self.images)})
        
        images_cleaned = []
        # Update progress bar after processing each image
        for count, img in enumerate(self.images, start=1):
            images_cleaned.append(img)
            self.master.after(0, lambda val=count: self.progress_bar.configure(value=val))
        
        if not images_cleaned:
            self.master.after(0, self.dimension_label.config, {"text": "Error: No valid images after processing."})
            return

        # Focus stacking using Laplacian variance
        stacked_image, focus_indices = focus_stack(images_cleaned)
        # Create depth map (using 0.05 mm per layer)
        depth_map = create_depth_map(focus_indices, layer_distance=0.05)
        # Generate point cloud with continuous height-based color mapping
        point_cloud, colors = depth_map_to_point_cloud(depth_map, focus_indices, num_images=len(self.images),
                                                        xy_scale=0.01, z_scale=1.0)
        # Calculate dimensions for display
        points_np = point_cloud  # already a numpy array
        length, breadth, height = calculate_dimensions(points_np)
        dim_text = f"Length: {length:.2f} mm, Breadth: {breadth:.2f} mm, Height: {height:.2f} mm"
        self.master.after(0, self.dimension_label.config, {"text": dim_text})
        
        gc.collect()
        
        # Visualize the point cloud using Open3D
        self.visualize_point_cloud(point_cloud, colors)

    def visualize_point_cloud(self, points, colors):
        if points.size == 0:
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
