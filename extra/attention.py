import sys
import cv2
import numpy as np
import glob
import torch
import threading
import torchvision.models as models
from torchvision import transforms as T
from tkinter import Tk, Label, Button, filedialog, messagebox, Frame
from tkinter import ttk
import open3d as o3d
import os

# -------------------- Custom Attention Segmentation Model --------------------
import torch.nn as nn

class AttentionModule(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(AttentionModule, self).__init__()
        self.W_g = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.W_x = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(torch.relu(g1 + x1))
        return x * psi

class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoder, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decode(x)

class AttentionSegmentationModel(nn.Module):
    def __init__(self, out_channels=1):
        super(AttentionSegmentationModel, self).__init__()
        backbone = models.resnet18(pretrained=True)
        self.base = nn.Sequential(*list(backbone.children())[:-2])
        self.attn = AttentionModule(in_channels=512, inter_channels=256)
        self.decoder = UNetDecoder(in_channels=1024, out_channels=out_channels)

    def forward(self, x):
        features = self.base(x)
        attn = self.attn(features, features)
        concat = torch.cat((features, attn), dim=1)
        out = self.decoder(concat)
        return out

# -------------------- Core Functions --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seg_model = AttentionSegmentationModel().to(device)
seg_model.eval()

def preprocess_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (224, 224))
    transform = T.Compose([T.ToTensor()])
    return transform(image_resized).unsqueeze(0).to(device)

def get_attention_mask(image):
    try:
        tensor_img = preprocess_image(image)
        with torch.no_grad():
            output = seg_model(tensor_img)
        mask = output.squeeze().cpu().numpy()
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        mask = (mask > 0.5).astype(np.uint8) * 255
        return mask
    except Exception as e:
        print(f"[Fallback] Attention segmentation failed: {e}")
        return traditional_masking(image)

def traditional_masking(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    mask = cv2.inRange(dilated_edges, 1, 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def mask_image_with_attention(image):
    mask = get_attention_mask(image)
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
    return x_max - x_min, y_max - y_min, z_max - z_min

def load_images_from_folder(folder_path):
    image_files = sorted(glob.glob(os.path.join(folder_path, '*.jpg')))
    return [cv2.imread(img_file) for img_file in image_files]

# -------------------- GUI Application --------------------
class PointCloudApp:
    def __init__(self, master):
        self.master = master
        self.master.title('3D Point Cloud Reconstruction Tool')
        self.master.geometry('800x600')
        self.frame = Frame(master)
        self.frame.pack()
        
        Label(self.frame, text="3D Point Cloud Reconstruction", font=("Arial", 18, "bold")).pack()
        Button(self.frame, text='Upload Images', command=self.upload_images).pack(pady=10)
        self.process_button = Button(self.frame, text='Process Images', command=self.start_processing)
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
    
    def start_processing(self):
        if not self.images:
            messagebox.showwarning("Warning", "Please upload images first.")
            return
        threading.Thread(target=self.run_point_cloud_processing, daemon=True).start()
    
    def run_point_cloud_processing(self):
        self.progress_bar["maximum"] = len(self.images)
        images_cleaned = []
        
        for i, img in enumerate(self.images):
            masked_img, _ = mask_image_with_attention(img)
            images_cleaned.append(masked_img)
            self.master.after(0, self.progress_bar.step, 1)
            
        if not images_cleaned:
            self.dimension_label.config(text="Error: No valid images after processing.")
            return
        
        stacked_image, focus_indices = focus_stack(images_cleaned)
        depth_map = create_depth_map(focus_indices, layer_distance=100)
        point_cloud, colors = depth_map_to_point_cloud(depth_map, stacked_image, xy_scale=0.01, z_scale=0.001)
        length, breadth, height = calculate_dimensions(point_cloud)
        
        self.master.after(0, self.dimension_label.config, {"text": f'Length: {length:.2f} mm, Breadth: {breadth:.2f} mm, Height: {height:.2f} mm'})
        self.visualize_point_cloud(point_cloud, colors)

    def visualize_point_cloud(self, points, colors):
        if len(points) == 0:
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    root = Tk()
    app = PointCloudApp(root)
    root.mainloop()
