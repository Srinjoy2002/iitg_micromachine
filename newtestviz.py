import sys
import cv2
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T
from tkinter import Tk, Label, Button, filedialog, messagebox, Frame
from tkinter import ttk
import open3d as o3d
import os

###############################################
#           Model Definition (Fast)           #
###############################################

# Attention Gate module (as in Attention U-Net)
class AttentionGate(nn.Module):
    def __init__(self, F_l, F_g, F_int):
        """
        F_l: Number of channels of the skip connection (from encoder)
        F_g: Number of channels of the gating signal (from decoder)
        F_int: Number of intermediate channels (typically F_l//2)
        """
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        """
        x: feature map from the encoder (skip connection)
        g: feature map from the decoder (gating signal)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        # Multiply attention coefficients back to the skip connection
        return x * psi

# Up-sampling block with attention (for the decoder)
class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        """
        in_channels: number of channels from the previous decoder level (gating signal)
        skip_channels: number of channels from the encoder skip connection
        out_channels: number of output channels after fusion and convolution
        """
        super(UpBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.attention = AttentionGate(F_l=skip_channels, F_g=in_channels, F_int=skip_channels // 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip):
        x = self.upsample(x)
        if x.size()[2:] != skip.size()[2:]:
            x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=True)
        skip = self.attention(skip, x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

# Lighter segmentation network using ResNet-18 as encoder
class ResNet18AttentionUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, pretrained=True):
        """
        n_channels: number of input channels (3 for RGB)
        n_classes: number of output channels (1 for binary mask)
        pretrained: if True, load a pretrained ResNet-18 encoder
        """
        super(ResNet18AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Use a pretrained ResNet-18 as encoder
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        # Initial block: conv1, bn1, relu -> output: 64 channels, H/2, W/2
        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.maxpool = resnet.maxpool  # Reduces resolution to H/4, W/4
        self.encoder1 = resnet.layer1   # 64 channels, H/4, W/4
        self.encoder2 = resnet.layer2   # 128 channels, H/8, W/8
        self.encoder3 = resnet.layer3   # 256 channels, H/16, W/16
        self.encoder4 = resnet.layer4   # 512 channels, H/32, W/32
        
        # Decoder with four upsampling stages.
        self.up1 = UpBlock(in_channels=512, skip_channels=256, out_channels=256)  # using encoder3 skip
        self.up2 = UpBlock(in_channels=256, skip_channels=128, out_channels=128)  # using encoder2 skip
        self.up3 = UpBlock(in_channels=128, skip_channels=64, out_channels=64)    # using encoder1 skip
        self.up4 = UpBlock(in_channels=64, skip_channels=64, out_channels=64)     # using initial block skip
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x0 = self.initial(x)         # (64, H/2, W/2)
        x1 = self.maxpool(x0)        # (64, H/4, W/4)
        x1 = self.encoder1(x1)       # (64, H/4, W/4)
        x2 = self.encoder2(x1)       # (128, H/8, W/8)
        x3 = self.encoder3(x2)       # (256, H/16, W/16)
        x4 = self.encoder4(x3)       # (512, H/32, W/32)
        
        # Decoder with attention in skip connections
        d1 = self.up1(x4, x3)        # (256, H/16, W/16)
        d2 = self.up2(d1, x2)        # (128, H/8, W/8)
        d3 = self.up3(d2, x1)        # (64, H/4, W/4)
        d4 = self.up4(d3, x0)        # (64, H/2, W/2)
        out = self.final_upsample(d4)   # Upsample to original resolution
        out = self.final_conv(out)
        out = torch.sigmoid(out)     # For a binary mask
        return out

###############################################
#     Instantiate the Segmentation Model      #
###############################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Using the lighter ResNet18AttentionUNet for faster mask creation
segmentation_model = ResNet18AttentionUNet(n_channels=3, n_classes=1, pretrained=True)
segmentation_model.to(device)
segmentation_model.eval()

###############################################
#          Other Utility Functions            #
###############################################

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
    """
    Uses the ResNet18-Attention UNet model to generate a segmentation mask.
    Returns a binary mask.
    """
    with torch.no_grad():
        image_tensor = preprocess_image(image)
        output = segmentation_model(image_tensor)  # output shape: (1, 1, H, W)
    mask = output[0, 0].cpu().numpy() * 255
    mask = mask.astype(np.uint8)
    # Threshold to obtain a binary mask
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary_mask

def traditional_masking(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    mask = cv2.inRange(dilated_edges, 1, 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def mask_image_with_resnet_attention(image):
    """
    Uses the ResNet18-Attention UNet for masking.
    Falls back to traditional masking if an exception occurs.
    """
    try:
        mask = get_object_mask(image)
    except Exception as e:
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

###############################################
#            Tkinter Application              #
###############################################

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
            masked_img, mask = mask_image_with_resnet_attention(img)
            if masked_img is not None:
                images_cleaned.append(masked_img)
            self.progress_bar['value'] = i + 1
            self.master.update_idletasks()

        if len(images_cleaned) == 0:
            self.dimension_label.config(text="Error: No valid images after processing.")
            return

        # Focus stacking
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
        pcd = pcd.voxel_down_sample(voxel_size=0.001)
        o3d.visualization.draw_geometries([pcd], window_name="Point Cloud", width=800, height=600)

###############################################
#                  Main                       #
###############################################

if __name__ == "__main__":
    root = Tk()
    app = PointCloudApp(root)
    root.mainloop()
