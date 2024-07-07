#import necessary packages
import sys
import cv2
import numpy as np
import glob
import open3d as o3d
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QLineEdit, QFormLayout, QMessageBox, QProgressBar
from PyQt5.QtGui import QFont, QPixmap, QIcon, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint

#Loading images from a folder

#The load_images_from_folder function takes a folder path as input and returns a list of images loaded from the folder using OpenCV.
def load_images_from_folder(folder_path):
    image_files = glob.glob(folder_path + '/*.jpg')
    images = [cv2.imread(img_file) for img_file in image_files]
    return images


def remove_background(image, lower_color, upper_color):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)

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

        self.image = None
        self.image_path = None
        self.scaled_image = None
        self.start_point = None
        self.end_point = None
        self.drawing = False

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Point Cloud and Dimensions')
        self.setGeometry(100, 100, 1000, 800)

        layout = QVBoxLayout()

        # Header with image
        header_layout = QVBoxLayout()
        header_image_path = 'images.png'  # Replace with your image path
        header_image_label = QLabel(self)
        pixmap = QPixmap(header_image_path)
        header_image_label.setPixmap(pixmap.scaledToWidth(100))  # Adjust size as needed
        header_image_label.setAlignment(Qt.AlignRight)
        header_layout.addWidget(header_image_label)

        header_title_label = QLabel('Advanced Point Cloud Reconstruction and Visualisation Tool', self)
        header_title_label.setFont(QFont('Arial', 18, QFont.Bold))
        header_title_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(header_title_label)

        layout.addLayout(header_layout)

        # Image display area
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(800, 600)
        layout.addWidget(self.image_label)

        # Load image button
        self.load_image_button = QPushButton('Load Image', self)
        self.load_image_button.clicked.connect(self.load_image)
        layout.addWidget(self.load_image_button)

        # Dimension display label
        self.dimension_label = QLabel(self)
        layout.addWidget(self.dimension_label)

        # Form layout for inputs
        form_layout = QFormLayout()

        # Input fields
        self.folder_input = QLineEdit(self)
        self.xy_scale_input = QLineEdit(self)
        self.z_scale_input = QLineEdit(self)

        self.folder_input.setPlaceholderText('Select folder containing images')
        self.xy_scale_input.setPlaceholderText('Enter XY scale factor in mm')
        self.z_scale_input.setPlaceholderText('Enter Z scale micron level')

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
        self.point_cloud_dimension_label = QLabel(self)
        self.point_cloud_dimension_label.setFont(QFont('Arial', 12))
        self.point_cloud_dimension_label.setAlignment(Qt.AlignCenter)

        # Adding widgets to form layout
        form_layout.addRow('Image Folder:', self.folder_input)
        form_layout.addWidget(browse_button)
        form_layout.addRow('XY Scale factor(mm):', self.xy_scale_input)
        form_layout.addRow('Z Scale micron level(mm):', self.z_scale_input) ## enter the micron level at which the image is taken
        form_layout.addWidget(start_button)

        # Adding form layout, progress bar, and dimension label to main layout
        layout.addLayout(form_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.point_cloud_dimension_label)

        # Set the main layout
        self.setLayout(layout)
        self.show()

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Image Files (*.png *.jpg *.bmp)')
        if file_path:
            self.image_path = file_path
            print(f"Selected image path: {self.image_path}")

            self.image = cv2.imread(self.image_path)
            if self.image is None:
                QMessageBox.critical(self, 'Error', f"Could not open or read the image file: {self.image_path}")
                return
            self.display_image()

    def display_image(self):
        if self.image is None:
            return

        # Convert the image to QImage for display
        height, width, channel = self.image.shape
        bytes_per_line = 3 * width
        q_image = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.scaled_image = q_image.scaled(self.image_label.size(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(QPixmap.fromImage(self.scaled_image))

    def paintEvent(self, event):
        super().paintEvent(event)

        if self.image is not None:
            painter = QPainter(self)
            painter.drawImage(self.image_label.pos(), self.scaled_image)

            if self.start_point and self.end_point:  ##marker to mark
                pen = QPen(Qt.white, 3, Qt.SolidLine)
                painter.setPen(pen)
                painter.drawLine(self.start_point, self.end_point)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.image is not None:
            self.start_point = event.pos() - self.image_label.pos()
            self.end_point = self.start_point
            self.drawing = True
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_point = event.pos() - self.image_label.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.end_point = event.pos() - self.image_label.pos()
            self.drawing = False
            self.update()

            # Calculate the distance in pixels and convert to mm
            self.calculate_distance()

    def calculate_distance(self):
        if self.start_point and self.end_point:
            dx = self.end_point.x() - self.start_point.x()
            dy = self.end_point.y() - self.start_point.y()
            distance_pixels = np.sqrt(dx ** 2 + dy ** 2)

            # Calculate the pixel to mm scale based on the displayed image size and original image size
            original_height, original_width = self.image.shape[:2]
            displayed_height = self.image_label.pixmap().height()
            displayed_width = self.image_label.pixmap().width()
            x_scale = original_width / displayed_width
            y_scale = original_height / displayed_height

            # Use the average of the x and y scales for simplicity
            pixel_to_mm_scale = (x_scale + y_scale) / 2

            distance_scale = distance_pixels * pixel_to_mm_scale/50

            self.dimension_label.setText(f'Length of marked region: {(25.4/100)*distance_scale:.2f} mm') #1 pixel=0.25 at dpi MM=P/(DPI/25.4)

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
            xy_scale = float(self.xy_scale_input.text())/100
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

            point_cloud = depth_map_to_point_cloud(depth_map, xy_scale=xy_scale, z_scale=z_scale)
            length, breadth, height = calculate_dimensions(point_cloud)

            # self.point_cloud_dimension_label.setText(f'Length: {length:.2f} mm, Breadth: {breadth:.2f} mm, Height: {height:.2f} mm')

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
