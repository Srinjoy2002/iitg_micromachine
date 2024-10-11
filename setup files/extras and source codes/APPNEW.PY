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

        # Check if predictions contain masks and visualize
        if 'masks' not in prediction or len(prediction['masks']) == 0:
            raise ValueError("No valid masks detected")

        # Print detection details for debugging
        print("Prediction Classes: ", prediction['labels'])
        print("Prediction Scores: ", prediction['scores'])

        # Only proceed if the highest score is above a threshold (adjust the threshold if needed)
        threshold = 0.01
        if prediction['scores'][0] < threshold:
            raise ValueError("No masks with a score above threshold")

        # Get the mask with the highest score
        mask = prediction['masks'][0, 0].mul(255).byte().cpu().numpy()
        return mask, prediction  # Also return prediction for visualization

# Load images from folder and verify if loaded correctly
def load_images_from_folder(folder_path):
    image_files = glob.glob(folder_path + '/*.jpg')
    if not image_files:
        raise FileNotFoundError(f"No images found in the folder: {folder_path}")
    images = [cv2.imread(img_file) for img_file in image_files]
    for i, img in enumerate(images):
        if img is None:
            print(f"Error reading image {image_files[i]}")
    return images

# Visualize detection results (bounding boxes and masks)
def visualize_detection(image, prediction):
    for i in range(len(prediction['boxes'])):
        box = prediction['boxes'][i].cpu().numpy().astype(int)
        score = prediction['scores'][i].cpu().numpy()

        # Only visualize if score is above the threshold
        if score >= 0.5:
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # Convert image back to QImage and display
    height, width, channel = image.shape
    bytes_per_line = 3 * width
    q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
    return QPixmap.fromImage(q_image)

# Function to mask the image using the Mask R-CNN
def mask_image_with_rcnn(image):
    try:
        mask, prediction = get_object_mask(image)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        return masked_image, prediction
    except ValueError as e:
        print(f"Skipping image due to error: {e}")
        return image, None

# Main GUI application
class PointCloudApp(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.initUI()

    def initUI(self):
        # Set window title and maximize the window
        self.setWindowTitle('Advanced 3D Point Cloud Reconstruction Tool')
        self.showMaximized()  # Maximize the window

        # Main layout for the window
        layout = QVBoxLayout()

        # Top layout for the logo and title
        top_layout = QHBoxLayout()

        # Logo section
        logo_label = QLabel(self)
        pixmap = QPixmap("logo.png")  # Add your logo file path here
        logo_label.setPixmap(pixmap.scaled(150, 150, Qt.KeepAspectRatio))  # Increase logo size
        top_layout.addWidget(logo_label, alignment=Qt.AlignLeft)

        # Title section
        title_label = QLabel("Advanced 3D Point Cloud Reconstruction Tool", self)
        title_label.setFont(QFont("Arial", 24, QFont.Bold))  # Larger and bold font for title
        top_layout.addWidget(title_label, alignment=Qt.AlignCenter)

        # Progress bar and status label
        self.dimension_label = QLabel(self)
        self.dimension_label.setFont(QFont("Arial", 12))  # Font for status and dimension text
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #05B8CC; }")  # Styling the progress bar

        # Add top layout and the rest of the GUI elements to the main layout
        layout.addLayout(top_layout)
        layout.addWidget(self.dimension_label)
        layout.addWidget(self.progress_bar)

        # Input for z_scale
        self.z_scale_input = QLineEdit(self)
        self.z_scale_input.setPlaceholderText("Enter z_scale (e.g., 0.001)")
        layout.addWidget(self.z_scale_input)

        # Button for uploading images
        self.upload_button = QPushButton('Upload Images', self)
        self.upload_button.setStyleSheet("background-color: #007ACC; color: white; font-weight: bold;")
        self.upload_button.clicked.connect(self.upload_images)
        layout.addWidget(self.upload_button)

        # Add a button for processing
        self.process_button = QPushButton('Process Images', self)
        self.process_button.setStyleSheet("background-color: #28A745; color: white; font-weight: bold;")
        self.process_button.clicked.connect(self.run_point_cloud_processing)
        layout.addWidget(self.process_button)

        # Set the main layout
        self.setLayout(layout)

    def upload_images(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.images = load_images_from_folder(folder)
            self.dimension_label.setText(f"{len(self.images)} images loaded.")

    def run_point_cloud_processing(self):
        if not hasattr(self, 'images'):
            self.dimension_label.setText("Please upload images first.")
            return

        self.progress_bar.setMaximum(len(self.images))

        images_cleaned = []
        for i, img in enumerate(self.images):
            masked_img, prediction = mask_image_with_rcnn(img)
            if prediction:
                # Visualize detection results
                pixmap = visualize_detection(img, prediction)
                self.dimension_label.setPixmap(pixmap)  # Display the image with bounding boxes and masks
            images_cleaned.append(masked_img)
            self.progress_bar.setValue(i + 1)

        if len(images_cleaned) == 0:
            self.dimension_label.setText("Error: No valid images after Mask R-CNN processing.")
            return

        stacked_image, focus_indices = focus_stack(images_cleaned)
        depth_map = create_depth_map(focus_indices, layer_distance=100)

        pixel_to_mm_scale = 0.01
        z_scale = float(self.z_scale_input.text() or 0.001)

        point_cloud, colors = depth_map_to_point_cloud(depth_map, stacked_image, xy_scale=pixel_to_mm_scale, z_scale=z_scale)
        length, breadth, height = calculate_dimensions(point_cloud)

        self.dimension_label.setText(f'Length: {length:.2f} mm, Breadth: {breadth:.2f} mm, Height: {height:.2f} mm')

        self.visualize_point_cloud(point_cloud, colors)

    def visualize_point_cloud(self, points, colors):
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

        o3d.visualization.draw_geometries([pcd, coordinate_frame], 
                                          zoom=0.5, 
                                          front=[0.0, 0.0, -1.0],
                                          lookat=[0.0, 0.0, 0.0],
                                          up=[0.0, -1.0, 0.0])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = PointCloudApp()
    sys.exit(app.exec_())
