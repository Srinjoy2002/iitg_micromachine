import sys
import cv2
import numpy as np
import glob
import open3d as o3d
import os
import logging
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, 
                             QPushButton, QFileDialog, QLineEdit, QProgressBar, QHBoxLayout)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision import transforms as T

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Mask R-CNN model (Pretrained)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model.to(device)
    logging.info("Mask R-CNN model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    sys.exit(1)

def load_images_from_folder(folder_path):
    """Load images from a specified folder."""
    image_files = glob.glob(folder_path + '/*.jpg')
    if not image_files:
        raise FileNotFoundError(f"No images found in the folder: {folder_path}")
    return [cv2.imread(img_file) for img_file in image_files]

def preprocess_image(image):
    """Preprocess image for Mask R-CNN."""
    transform = T.Compose([T.ToTensor()])
    return transform(image).unsqueeze(0).to(device)

def get_object_mask(image):
    """Obtain object mask from the image using Mask R-CNN."""
    try:
        with torch.no_grad():
            image_tensor = preprocess_image(image)
            prediction = model(image_tensor)[0]
            if 'masks' not in prediction or len(prediction['masks']) == 0:
                logging.warning("No valid masks detected.")
                return None
            return prediction['masks'][0, 0].mul(255).byte().cpu().numpy()
    except Exception as e:
        logging.error(f"Error during mask extraction: {e}")
        return None

def mask_image_with_rcnn(image):
    """Mask the image using the object mask."""
    mask = get_object_mask(image)
    if mask is not None:
        return cv2.bitwise_and(image, image, mask=mask)
    else:
        logging.warning("Using traditional masking method.")
        return apply_traditional_masking(image)

def apply_traditional_masking(image):
    """Fallback method to apply traditional masking using color thresholds."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, (255, 255, 255), -1)
        return cv2.bitwise_and(image, mask)
    return image

def focus_stack(images):
    """Stack only in-focus images based on variance."""
    focus_measure = [cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() for img in images]
    focus_threshold = np.mean(focus_measure)  # Use mean variance as a threshold

    stacked_image = np.zeros(images[0].shape, dtype=images[0].dtype)
    for idx, measure in enumerate(focus_measure):
        if measure > focus_threshold:
            stacked_image += images[idx]

    return np.clip(stacked_image, 0, 255).astype(np.uint8)

def create_depth_map(focus_indices, layer_distance):
    """Create a depth map from focus indices."""
    return focus_indices * layer_distance

def depth_map_to_point_cloud(depth_map, image, xy_scale=1.0, z_scale=1.0):
    """Convert depth map to 3D point cloud."""
    h, w = depth_map.shape
    points = []
    colors = []

    for y in range(h):
        for x in range(w):
            z = depth_map[y, x] * z_scale
            points.append([x * xy_scale, y * xy_scale, z])
            colors.append(image[y, x] / 255.0)

    return np.array(points), np.array(colors)

def calculate_dimensions(points):
    """Calculate the dimensions of the point cloud."""
    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)
    return x_max - x_min, y_max - y_min, z_max - z_min

class PointCloudApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Advanced 3D Point Cloud Reconstruction Tool')
        self.showMaximized()

        layout = QVBoxLayout()

        # Top layout for logo and title
        top_layout = QHBoxLayout()
        logo_label = QLabel(self)
        pixmap = QPixmap("logo.png")  # Your logo path
        logo_label.setPixmap(pixmap.scaled(150, 150, Qt.KeepAspectRatio))
        top_layout.addWidget(logo_label, alignment=Qt.AlignLeft)

        title_label = QLabel("Advanced 3D Point Cloud Reconstruction Tool", self)
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        top_layout.addWidget(title_label, alignment=Qt.AlignCenter)

        layout.addLayout(top_layout)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #05B8CC; }")
        layout.addWidget(self.progress_bar)

        self.dimension_label = QLabel(self)
        self.dimension_label.setFont(QFont("Arial", 12))
        layout.addWidget(self.dimension_label)

        self.folder_input = QLineEdit(self)
        self.folder_input.setPlaceholderText("Select folder containing images")
        layout.addWidget(self.folder_input)

        self.browse_button = QPushButton('Browse Folder', self)
        self.browse_button.setStyleSheet("background-color: #007ACC; color: white; font-weight: bold;")
        self.browse_button.clicked.connect(self.browse_folder)
        layout.addWidget(self.browse_button)

        self.process_button = QPushButton('Start Processing', self)
        self.process_button.setStyleSheet("background-color: #28A745; color: white; font-weight: bold;")
        self.process_button.clicked.connect(self.run_point_cloud_processing)
        layout.addWidget(self.process_button)

        self.setLayout(layout)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.folder_input.setText(folder)

    def run_point_cloud_processing(self):
        folder = self.folder_input.text()
        if not folder or not os.path.exists(folder):
            self.dimension_label.setText("Please select a valid folder.")
            return

        images = load_images_from_folder(folder)
        if not images:
            self.dimension_label.setText("No images found in the selected folder.")
            return
        
        lower_color = np.array([0, 0, 200])
        upper_color = np.array([180, 255, 255])

        images_cleaned = []
        total_images = len(images)
        for i, img in enumerate(images):
            masked_img = mask_image_with_rcnn(img)
            images_cleaned.append(masked_img)

            # Update progress bar
            self.progress_bar.setValue(int((i + 1) / total_images * 100))
            logging.debug(f"Processed image {i + 1}/{total_images}")

        if not images_cleaned:
            self.dimension_label.setText("Error: No valid images after processing.")
            return

        stacked_image = focus_stack(images_cleaned)
        depth_map = create_depth_map(np.arange(len(images_cleaned)), layer_distance=100)

        pixel_to_mm_scale = 0.01
        z_scale = 0.001

        point_cloud, colors = depth_map_to_point_cloud(depth_map, stacked_image, xy_scale=pixel_to_mm_scale, z_scale=z_scale)
        length, breadth, height = calculate_dimensions(point_cloud)

        self.dimension_label.setText(f'Length: {length:.2f} mm, Breadth: {breadth:.2f} mm, Height: {height:.2f} mm')
        self.visualize_point_cloud(point_cloud)

    def visualize_point_cloud(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, coordinate_frame])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = PointCloudApp()
    sys.exit(app.exec_())
