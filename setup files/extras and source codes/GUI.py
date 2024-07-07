import sys
import cv2
import numpy as np
import glob
import open3d as o3d
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QLineEdit, QFormLayout, QMessageBox, QProgressBar
from PyQt5.QtGui import QFont, QPixmap, QIcon
from PyQt5.QtCore import Qt

def load_images_from_folder(folder_path):
    image_files = glob.glob(folder_path + '/*.jpg')
    images = [cv2.imread(img_file) for img_file in image_files]
    return images

def remove_background(image, lower_color, upper_color):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Apply morphological operations to remove small noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def focus_stack(images):
    stack_shape = images[0].shape[:2]
    focus_measure = np.zeros(stack_shape)
    focus_indices = np.zeros(stack_shape, dtype=int)

    for i, image in enumerate(images):
        laplacian = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
        mask = laplacian > focus_measure
        focus_measure[mask] = laplacian[mask]
        focus_indices[mask] = i

    stacked_image = np.zeros(images[0].shape, dtype=images[0].dtype)
    for y in range(stack_shape[0]):
        for x in range(stack_shape[1]):
            stacked_image[y, x] = images[focus_indices[y, x]][y, x]

    return stacked_image, focus_indices

def create_depth_map(focus_indices, layer_distance):
    depth_map = focus_indices * layer_distance
    return depth_map

def depth_map_to_point_cloud(depth_map, xy_scale=1.0, z_scale=1.0):
    h, w = depth_map.shape
    points = []

    for y in range(h):
        for x in range(w):
            z = depth_map[y, x] * z_scale
            points.append([x * xy_scale, y * xy_scale, z])

    return np.array(points)

def calculate_dimensions(points):
    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)
    length = x_max - x_min
    breadth = y_max - y_min
    height = z_max - z_min
    return length, breadth, height

class PointCloudApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Point Cloud and Dimensions')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        # Load heading image
        heading_image_path = "C:/Users/david/OneDrive/Desktop/sri/cap/images.png"
        heading_label = QLabel(self)
        pixmap = QPixmap(heading_image_path)
        heading_label.setPixmap(pixmap)
        heading_label.setAlignment(Qt.AlignRight)
        layout.addWidget(heading_label)

        # Title label
        title_label = QLabel('Advanced Point Cloud Reconstruction Tool', self)
        title_label.setFont(QFont('Arial', 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Form layout for inputs
        form_layout = QFormLayout()

        # Input fields
        self.folder_input = QLineEdit(self)
        self.xy_scale_input = QLineEdit(self)
        self.z_scale_input = QLineEdit(self)

        self.folder_input.setPlaceholderText('Select folder containing images')
        self.xy_scale_input.setPlaceholderText('Enter XY scale (e.g., 0.01)')
        self.z_scale_input.setPlaceholderText('Enter Z scale (e.g., 0.001)')

        # Browse button
        browse_button = QPushButton('Browse', self)
        browse_button.clicked.connect(self.browse_folder)

        # Start button
        start_button = QPushButton('Start Processing', self)
        start_button.clicked.connect(self.run_point_cloud_processing)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMaximum(100)

        # Dimension label
        self.dimension_label = QLabel(self)
        self.dimension_label.setFont(QFont('Arial', 12))
        self.dimension_label.setAlignment(Qt.AlignCenter)

        # Adding widgets to form layout
        form_layout.addRow('Image Folder:', self.folder_input)
        form_layout.addWidget(browse_button)
        form_layout.addRow('XY Scale:', self.xy_scale_input)
        form_layout.addRow('Z Scale:', self.z_scale_input)
        form_layout.addWidget(start_button)

        # Adding form layout, progress bar, and dimension label to main layout
        layout.addLayout(form_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.dimension_label)

        # Set the main layout
        self.setLayout(layout)
        self.show()

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if folder:
            self.folder_input.setText(folder)

    def run_point_cloud_processing(self):
        folder = self.folder_input.text()
        if not folder or not os.path.exists(folder):
            QMessageBox.warning(self, 'Error', 'Please select a valid folder containing images.')
            return

        try:
            xy_scale = float(self.xy_scale_input.text())
            z_scale = float(self.z_scale_input.text())
        except ValueError:
            QMessageBox.warning(self, 'Error', 'Please enter valid numeric values for XY scale and Z scale.')
            return

        try:
            images = load_images_from_folder(folder)
            if not images:
                QMessageBox.warning(self, 'Error', 'No images found in the selected folder.')
                return

            images_cleaned = [remove_background(img, lower_color=np.array([0, 0, 200]), upper_color=np.array([180, 255, 255])) for img in images]

            # Save cleaned images to a new folder
            cleaned_folder = 'cleaned_images'
            if not os.path.exists(cleaned_folder):
                os.makedirs(cleaned_folder)
            for i, img in enumerate(images_cleaned):
                cv2.imwrite(os.path.join(cleaned_folder, f'cleaned_{i}.jpg'), img)

            stacked_image, focus_indices = focus_stack(images_cleaned)
            depth_map = create_depth_map(focus_indices, layer_distance=100)  # 100 microns

            # Assuming the object is 2mm by 2mm and images are 200x200 pixels
            pixel_to_mm_scale = 0.01
            z_scale = 0.001

            point_cloud = depth_map_to_point_cloud(depth_map, xy_scale=pixel_to_mm_scale, z_scale=z_scale)
            length, breadth, height = calculate_dimensions(point_cloud)

            self.dimension_label.setText(f'Length: {length:.2f} mm, Breadth: {breadth:.2f} mm, Height: {height:.2f} mm')

            self.visualize_point_cloud(point_cloud)

        except Exception as e:
            QMessageBox.critical(self, 'Error', f'An error occurred: {e}')
            raise e

    def visualize_point_cloud(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, coordinate_frame])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = PointCloudApp()
    sys.exit(app.exec_())
