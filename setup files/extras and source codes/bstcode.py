import sys
import cv2
import numpy as np
import glob
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision import transforms as T
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, QProgressBar, 
                             QPushButton, QLineEdit, QFileDialog, QHBoxLayout)
from PyQt5.QtGui import QPixmap, QFont, QImage
from PyQt5.QtCore import Qt
import os
import gc

# Load Mask R-CNN model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
model.to(device)

# Preprocessing function for images
def preprocess_image(image):
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.ToTensor()])
    return transform(image_rgb).unsqueeze(0).to(device)

# Function to get mask from Mask R-CNN
def get_object_mask(image):
    with torch.no_grad():
        image_tensor = preprocess_image(image)
        prediction = model(image_tensor)[0]

        # Check if predictions contain masks
        if 'masks' not in prediction or len(prediction['masks']) == 0:
            raise ValueError("No valid masks detected")

        # Print detection details for debugging
        print("Prediction Classes: ", prediction['labels'])
        print("Prediction Scores: ", prediction['scores'])

        threshold = 0.5  # Adjust the threshold if needed
        if prediction['scores'][0] < threshold:
            raise ValueError("No masks with a score above threshold")

        # Get the mask with the highest score
        mask = prediction['masks'][0, 0].mul(255).byte().cpu().numpy()
        return mask, prediction

# Load images from folder and verify if loaded correctly
def load_images_from_folder(folder_path):
    image_files = glob.glob(os.path.join(folder_path, '*.jpg'))
    if not image_files:
        raise FileNotFoundError(f"No images found in the folder: {folder_path}")
    images = [cv2.imread(img_file) for img_file in image_files]
    return images

# Visualize detection results (bounding boxes and masks)
def visualize_detection(image, prediction):
    image_copy = image.copy()
    for i in range(len(prediction['boxes'])):
        box = prediction['boxes'][i].cpu().numpy().astype(int)
        score = prediction['scores'][i].cpu().numpy()
        if score >= 0.5:
            cv2.rectangle(image_copy, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    height, width, channel = image_copy.shape
    bytes_per_line = 3 * width
    q_image = QImage(image_copy.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
    return QPixmap.fromImage(q_image)

# Function to mask the image using the Mask R-CNN
def mask_image_with_rcnn(image):
    try:
        mask, prediction = get_object_mask(image)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        return masked_image, mask, prediction
    except ValueError as e:
        print(f"Skipping image due to error: {e}")
        return None, None, None

# Focus stacking function
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

# Create depth map from focus indices
def create_depth_map(focus_indices, layer_distance):
    depth_map = focus_indices * layer_distance
    return depth_map

# Convert depth map to point cloud
def depth_map_to_point_cloud(depth_map, image, xy_scale=1.0, z_scale=1.0):
    h, w = depth_map.shape
    points = []
    colors = []

    for y in range(h):
        for x in range(w):
            z = depth_map[y, x] * z_scale
            if z != 0:  # Only add points with valid z
                points.append([x * xy_scale, y * xy_scale, z])
                colors.append(image[y, x] / 255.0)

    return np.array(points), np.array(colors)

# Calculate dimensions from point cloud
def calculate_dimensions(points):
    if len(points) == 0:
        return 0, 0, 0
    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)
    length = x_max - x_min
    breadth = y_max - y_min
    height = z_max - z_min
    return length, breadth, height

# Main GUI application
class PointCloudApp(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Advanced 3D Point Cloud Reconstruction Tool')
        self.showMaximized()

        layout = QVBoxLayout()
        top_layout = QHBoxLayout()

        logo_label = QLabel(self)
        pixmap = QPixmap("logo.png")  # Add your logo file path here
        logo_label.setPixmap(pixmap.scaled(150, 150, Qt.KeepAspectRatio))
        top_layout.addWidget(logo_label, alignment=Qt.AlignLeft)

        title_label = QLabel("Advanced 3D Point Cloud Reconstruction Tool", self)
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        top_layout.addWidget(title_label, alignment=Qt.AlignCenter)

        self.dimension_label = QLabel(self)
        self.dimension_label.setFont(QFont("Arial", 12))
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #05B8CC; }")

        layout.addLayout(top_layout)
        layout.addWidget(self.dimension_label)
        layout.addWidget(self.progress_bar)

        self.z_scale_input = QLineEdit(self)
        self.z_scale_input.setPlaceholderText("Enter z_scale (e.g., 0.001)")
        layout.addWidget(self.z_scale_input)

        self.upload_button = QPushButton('Upload Images', self)
        self.upload_button.setStyleSheet("background-color: #007ACC; color: white; font-weight: bold;")
        self.upload_button.clicked.connect(self.upload_images)
        layout.addWidget(self.upload_button)

        self.process_button = QPushButton('Process Images', self)
        self.process_button.setStyleSheet("background-color: #28A745; color: white; font-weight: bold;")
        self.process_button.clicked.connect(self.run_point_cloud_processing)
        layout.addWidget(self.process_button)

        self.setLayout(layout)

    def upload_images(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.image_folder = folder  # Save the folder path
            self.images = load_images_from_folder(folder)
            self.dimension_label.setText(f"{len(self.images)} images loaded.")

    def run_point_cloud_processing(self):
        if not hasattr(self, 'images'):
            self.dimension_label.setText("Please upload images first.")
            return

        self.progress_bar.setMaximum(len(self.images))

        images_cleaned = []
        masks = []
        for i, img in enumerate(self.images):
            masked_img, mask, prediction = mask_image_with_rcnn(img)
            if masked_img is not None and mask is not None:
                images_cleaned.append(masked_img)
                masks.append(mask)

                # Visualize detection results
                pixmap = visualize_detection(img, prediction)
                self.dimension_label.setPixmap(pixmap)  # Display the image with bounding boxes and masks
            else:
                # If mask is not obtained, use the original image
                images_cleaned.append(img)

            self.progress_bar.setValue(i + 1)

        if len(images_cleaned) == 0:
            self.dimension_label.setText("Error: No valid images after Mask R-CNN processing.")
            return

        # Save masks
        self.save_masks(masks)

        # Focus stacking
        stacked_image, focus_indices = focus_stack(images_cleaned)
        depth_map = create_depth_map(focus_indices, layer_distance=100)

        pixel_to_mm_scale = 0.01
        try:
            z_scale = float(self.z_scale_input.text() or 0.001)
        except ValueError:
            z_scale = 0.001

        point_cloud, colors = depth_map_to_point_cloud(depth_map, stacked_image, xy_scale=pixel_to_mm_scale, z_scale=z_scale)
        length, breadth, height = calculate_dimensions(point_cloud)

        self.dimension_label.setText(f'Length: {length:.2f} mm, Breadth: {breadth:.2f} mm, Height: {height:.2f} mm')

        self.visualize_point_cloud(point_cloud, colors)

    def save_masks(self, masks):
        # Create Masks folder with name based on image folder
        base_folder_name = os.path.basename(self.image_folder)
        mask_folder = os.path.join(os.getcwd(), f"Masks_{base_folder_name}")
        os.makedirs(mask_folder, exist_ok=True)
        for idx, mask in enumerate(masks):
            mask_filename = os.path.join(mask_folder, f"mask_{idx}.png")
            cv2.imwrite(mask_filename, mask)
        print(f"Masks saved to {mask_folder}")

    def visualize_point_cloud(self, points, colors):
        import open3d as o3d

        if len(points) == 0:
            self.dimension_label.setText("No points to display in the point cloud.")
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

        try:
            o3d.visualization.draw_geometries([pcd, coordinate_frame],
                                              zoom=0.5,
                                              front=[0.0, 0.0, -1.0],
                                              lookat=[0.0, 0.0, 0.0],
                                              up=[0.0, -1.0, 0.0])
        except Exception as e:
            print(f"Error in Open3D visualization: {e}")
            self.dimension_label.setText("Failed to open Open3D visualization window.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = PointCloudApp()
    sys.exit(app.exec_())


#best code till now