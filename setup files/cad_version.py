import sys
import cv2
import numpy as np
import glob
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision import transforms as T
from tkinter import Tk, Label, Button, filedialog, messagebox, Frame
from tkinter import ttk
import open3d as o3d
import os

# Load Mask R-CNN model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()
model.to(device)

def load_images_from_folder(folder_path):
    image_files = glob.glob(os.path.join(folder_path, '*.jpg'))
    if not image_files:
        raise FileNotFoundError(f"No images found in the folder: {folder_path}")
    images = [cv2.imread(img_file) for img_file in image_files]
    return images

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
        return mask

def traditional_masking(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    mask = cv2.inRange(dilated_edges, 1, 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def mask_image_with_rcnn(image):
    try:
        mask = get_object_mask(image)
    except ValueError:
        mask = traditional_masking(image)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image, mask

def focus_stack(images):
    stack_shape = images[0].shape[:2]
    focus_measure = np.zeros(stack_shape)
    focus_indices = np.zeros(stack_shape, dtype=int)

    for i, image in enumerate(images):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        mask = laplacian > focus_measure
        focus_measure[mask] = laplacian[mask]
        focus_indices[mask] = i

    stacked_image = np.zeros_like(images[0])
    for y in range(stack_shape[0]):
        for x in range(stack_shape[1]):
            stacked_image[y, x] = images[focus_indices[y, x]][y, x]

    return stacked_image, focus_indices

def create_depth_map(focus_indices, layer_distance):
    return focus_indices * layer_distance

def depth_map_to_point_cloud(depth_map, image, xy_scale=1.0, z_scale=1.0):
    h, w = depth_map.shape
    points = []
    colors = []

    for y in range(h):
        for x in range(w):
            z = depth_map[y, x] * z_scale
            if z != 0:
                points.append([x * xy_scale, y * xy_scale, z])
                color = image[y, x] / 255.0
                colors.append(color)

    return np.array(points), np.array(colors)

def calculate_dimensions(points):
    if len(points) == 0:
        return 0, 0, 0
    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)
    length = x_max - x_min
    breadth = y_max - y_min
    height = z_max - z_min
    return length, breadth, height

class PointCloudApp:
    def __init__(self, master):
        self.master = master
        self.master.title('3D Point Cloud Reconstruction Tool')
        self.master.geometry('800x600')

        self.frame = Frame(master)
        self.frame.pack()

        self.title_label = Label(self.frame, text="3D Point Cloud Reconstruction Tool", font=("Arial", 24, "bold"))
        self.title_label.pack()

        self.upload_button = Button(self.frame, text='Upload Images', command=self.upload_images)
        self.upload_button.pack(pady=10)

        self.process_button = Button(self.frame, text='Process Images', command=self.run_point_cloud_processing)
        self.process_button.pack(pady=10)

        self.dimension_label = Label(self.frame, text="")
        self.dimension_label.pack(pady=10)

        self.progress_bar = ttk.Progressbar(self.frame, length=300, mode='determinate')
        self.progress_bar.pack(pady=10)

        self.images = []

    def upload_images(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.images = load_images_from_folder(folder_path)
            messagebox.showinfo("Info", f"{len(self.images)} images loaded.")

    def run_point_cloud_processing(self):
        if not self.images:
            messagebox.showwarning("Warning", "Please upload images first.")
            return

        self.progress_bar['maximum'] = len(self.images)

        images_cleaned = []
        for i, img in enumerate(self.images):
            masked_img, mask = mask_image_with_rcnn(img)
            if masked_img is not None:
                images_cleaned.append(masked_img)

            progress = int((i + 1) / len(self.images) * 100)
            print(f"Progress: {progress}%")
            self.progress_bar['value'] = i + 1
            self.master.update_idletasks()

        if len(images_cleaned) == 0:
            self.dimension_label.config(text="Error: No valid images after processing.")
            return

        stacked_image, focus_indices = focus_stack(images_cleaned)
        depth_map = create_depth_map(focus_indices, layer_distance=100)

        pixel_to_mm_scale = 0.01
        point_cloud, colors = depth_map_to_point_cloud(depth_map, stacked_image, xy_scale=pixel_to_mm_scale, z_scale=0.001)
        length, breadth, height = calculate_dimensions(point_cloud)

        self.dimension_label.config(text=f'Length: {length:.2f} mm, Breadth: {breadth:.2f} mm, Height: {height:.2f} mm')
        self.visualize_point_cloud(point_cloud, colors)

    def visualize_point_cloud(self, points, colors):
        if len(points) == 0:
            self.dimension_label.config(text="No points to display in the point cloud.")
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(100)

        # Save the point cloud for CAD import
        o3d.io.write_point_cloud("point_cloud.ply", pcd)

        # Convert the point cloud to a mesh using Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        o3d.io.write_triangle_mesh("point_cloud.stl", mesh)

        o3d.visualization.draw_geometries([pcd], window_name="Point Cloud", width=800, height=600)

if __name__ == "__main__":
    root = Tk()
    app = PointCloudApp(root)
    root.mainloop()
