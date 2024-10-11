#original tested code

import sys
import cv2
import numpy as np
import glob
import open3d as o3d
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel

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

        self.dimension_label = QLabel(self)
        layout.addWidget(self.dimension_label)

        self.setLayout(layout)
        self.show()

        self.run_point_cloud_processing()

    def run_point_cloud_processing(self):
        profile_folders = ['data\nib side extended','D:\iitg\data\nib']
        
        # Define HSV color range for the object
        lower_color = np.array([0, 0, 200])  # Adjust these values based on your object color
        upper_color = np.array([180, 255, 255])

        all_point_clouds = []

        for folder in profile_folders:
            images = load_images_from_folder(folder)
            if not images:
                raise ValueError(f"Please provide at least two profile images in {folder}.")
                
            images_cleaned = [remove_background(img, lower_color, upper_color) for img in images]
            
            # Save cleaned images to a new folder
            cleaned_folder = 'cleaned_images'
            if not os.path.exists(cleaned_folder):
                os.makedirs(cleaned_folder)
            for i, img in enumerate(images_cleaned):
                cv2.imwrite(os.path.join(cleaned_folder, f'{folder.split()[0]}_{i}.jpg'), img)

            stacked_image, focus_indices = focus_stack(images_cleaned)
            depth_map = create_depth_map(focus_indices, layer_distance=100)  # 100 microns

            # Since the object is 2mm by 2mm, and assuming a 2mm x 2mm image size, we need a scaling factor.
            # Let's assume the images are 200x200 pixels, so each pixel represents 0.01mm.
            pixel_to_mm_scale = 0.01  # This would need adjusting if the images have a different resolution

            # Adjust the z-scale to make sure the point cloud is not too elongated
            z_scale = 0.001  # Hardcoded z-axis scaling factor to reduce elongation

            point_cloud = depth_map_to_point_cloud(depth_map, xy_scale=pixel_to_mm_scale, z_scale=z_scale)
            all_point_clouds.append(point_cloud)

        # Merge all point clouds
        merged_point_cloud = np.concatenate(all_point_clouds, axis=0)

        # Calculate dimensions
        length, breadth, height = calculate_dimensions(merged_point_cloud)

        self.dimension_label.setText(f'Length: {length:.2f} mm, Breadth: {breadth:.2f} mm, Height: {height:.2f} mm')

        # Visualize the point cloud
        self.visualize_point_cloud(merged_point_cloud)

    def visualize_point_cloud(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Create a coordinate frame for scale reference
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        
        # Visualize the point cloud with the coordinate frame
        o3d.visualization.draw_geometries([pcd, coordinate_frame])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = PointCloudApp()
    sys.exit(app.exec_())
# # # import sys
# # # import cv2
# # # import numpy as np
# # # import glob
# # # import open3d as o3d
# # # import os
# # # from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, 
# # #                              QPushButton, QFileDialog, QLineEdit, QProgressBar, QHBoxLayout)
# # # from PyQt5.QtGui import QPixmap, QFont
# # # from PyQt5.QtCore import Qt

# # # def load_images_from_folder(folder_path):
# # #     image_files = glob.glob(folder_path + '/*.jpg')
# # #     images = [cv2.imread(img_file) for img_file in image_files]
# # #     return images

# # # def remove_background(image, lower_color, upper_color):
# # #     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# # #     mask = cv2.inRange(hsv, lower_color, upper_color)
    
# # #     kernel = np.ones((5, 5), np.uint8)
# # #     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# # #     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
# # #     result = cv2.bitwise_and(image, image, mask=mask)
# # #     return result

# # # def variance_of_laplacian(image):
# # #     return cv2.Laplacian(image, cv2.CV_64F).var()

# # # def focus_stack(images):
# # #     stack_shape = images[0].shape[:2]
# # #     focus_measure = np.zeros(stack_shape)
# # #     focus_indices = np.zeros(stack_shape, dtype=int)

# # #     for i, image in enumerate(images):
# # #         laplacian = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
# # #         mask = laplacian > focus_measure
# # #         focus_measure[mask] = laplacian[mask]
# # #         focus_indices[mask] = i

# # #     stacked_image = np.zeros(images[0].shape, dtype=images[0].dtype)
# # #     for y in range(stack_shape[0]):
# # #         for x in range(stack_shape[1]):
# # #             stacked_image[y, x] = images[focus_indices[y, x]][y, x]

# # #     return stacked_image, focus_indices

# # # def create_depth_map(focus_indices, layer_distance):
# # #     depth_map = focus_indices * layer_distance
# # #     return depth_map

# # # def depth_map_to_point_cloud(depth_map, xy_scale=1.0, z_scale=1.0):
# # #     h, w = depth_map.shape
# # #     points = []

# # #     for y in range(h):
# # #         for x in range(w):
# # #             z = depth_map[y, x] * z_scale
# # #             points.append([x * xy_scale, y * xy_scale, z])

# # #     return np.array(points)

# # # def calculate_dimensions(points):
# # #     x_min, y_min, z_min = np.min(points, axis=0)
# # #     x_max, y_max, z_max = np.max(points, axis=0)
# # #     length = x_max - x_min
# # #     breadth = y_max - y_min
# # #     height = z_max - z_min
# # #     return length, breadth, height

# # # class PointCloudApp(QWidget):
# # #     def __init__(self):
# # #         super().__init__()
# # #         self.initUI()

# # #     def initUI(self):
# # #         self.setWindowTitle('Advanced 3D Point Cloud Reconstruction Tool')
# # #         self.showMaximized()

# # #         layout = QVBoxLayout()

# # #         # Top layout for logo and title
# # #         top_layout = QHBoxLayout()
# # #         # Logo (replace "logo.png" with your logo image path)
# # #         logo_label = QLabel(self)
# # #         pixmap = QPixmap("logo.png")
# # #         logo_label.setPixmap(pixmap.scaled(150, 150, Qt.KeepAspectRatio))
# # #         top_layout.addWidget(logo_label, alignment=Qt.AlignLeft)

# # #         # Title section
# # #         title_label = QLabel("Advanced 3D Point Cloud Reconstruction Tool", self)
# # #         title_label.setFont(QFont("Arial", 24, QFont.Bold))
# # #         top_layout.addWidget(title_label, alignment=Qt.AlignCenter)

# # #         layout.addLayout(top_layout)

# # #         # Progress bar
# # #         self.progress_bar = QProgressBar(self)
# # #         self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #05B8CC; }")
# # #         layout.addWidget(self.progress_bar)

# # #         # Dimension display label
# # #         self.dimension_label = QLabel(self)
# # #         self.dimension_label.setFont(QFont("Arial", 12))
# # #         layout.addWidget(self.dimension_label)

# # #         # Input for profile folder
# # #         self.folder_input = QLineEdit(self)
# # #         self.folder_input.setPlaceholderText("Select folder containing images")
# # #         layout.addWidget(self.folder_input)

# # #         # Browse button
# # #         self.browse_button = QPushButton('Browse Folder', self)
# # #         self.browse_button.setStyleSheet("background-color: #007ACC; color: white; font-weight: bold;")
# # #         self.browse_button.clicked.connect(self.browse_folder)
# # #         layout.addWidget(self.browse_button)

# # #         # Start button
# # #         self.process_button = QPushButton('Start Processing', self)
# # #         self.process_button.setStyleSheet("background-color: #28A745; color: white; font-weight: bold;")
# # #         self.process_button.clicked.connect(self.run_point_cloud_processing)
# # #         layout.addWidget(self.process_button)

# # #         # Set the main layout
# # #         self.setLayout(layout)

# # #     def browse_folder(self):
# # #         folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
# # #         if folder:
# # #             self.folder_input.setText(folder)

# # #     def run_point_cloud_processing(self):
# # #         folder = self.folder_input.text()
# # #         if not folder or not os.path.exists(folder):
# # #             self.dimension_label.setText("Please select a valid folder.")
# # #             return

# # #         profile_folders = [folder]
        
# # #         # Define HSV color range for the object
# # #         lower_color = np.array([0, 0, 200])
# # #         upper_color = np.array([180, 255, 255])

# # #         all_point_clouds = []

# # #         for folder in profile_folders:
# # #             images = load_images_from_folder(folder)
# # #             if not images:
# # #                 raise ValueError(f"Please provide at least two profile images in {folder}.")
                
# # #             images_cleaned = []
# # #             total_images = len(images)
# # #             for i, img in enumerate(images):
# # #                 cleaned_img = remove_background(img, lower_color, upper_color)
# # #                 images_cleaned.append(cleaned_img)

# # #                 # Update progress bar and terminal output
# # #                 progress = int(((i + 1) / total_images) * 100)
# # #                 self.progress_bar.setValue(progress)
# # #                 print(f"Processing image {i + 1}/{total_images} - {progress}% completed")

# # #             stacked_image, focus_indices = focus_stack(images_cleaned)
# # #             depth_map = create_depth_map(focus_indices, layer_distance=100)

# # #             pixel_to_mm_scale = 0.01
# # #             z_scale = 0.001

# # #             point_cloud = depth_map_to_point_cloud(depth_map, xy_scale=pixel_to_mm_scale, z_scale=z_scale)
# # #             all_point_clouds.append(point_cloud)

# # #         merged_point_cloud = np.concatenate(all_point_clouds, axis=0)

# # #         length, breadth, height = calculate_dimensions(merged_point_cloud)

# # #         self.dimension_label.setText(f'Length: {length:.2f} mm, Breadth: {breadth:.2f} mm, Height: {height:.2f} mm')

# # #         self.visualize_point_cloud(merged_point_cloud)

# # #     def visualize_point_cloud(self, points):
# # #         pcd = o3d.geometry.PointCloud()
# # #         pcd.points = o3d.utility.Vector3dVector(points)

# # #         coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
# # #         o3d.visualization.draw_geometries([pcd, coordinate_frame])

# # # if __name__ == "__main__":
# # #     app = QApplication(sys.argv)
# # #     ex = PointCloudApp()
# # #     sys.exit(app.exec_())



# import sys
# import cv2
# import numpy as np
# import glob
# import open3d as o3d
# import os
# from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, 
#                              QPushButton, QFileDialog, QLineEdit, QProgressBar, QHBoxLayout)
# from PyQt5.QtGui import QPixmap, QFont
# from PyQt5.QtCore import Qt

# # def load_images_from_folder(folder_path):
# #     supported_formats = ['*.jpg', '*.png', '*.jpeg', '*.bmp']
# #     image_files = []
# #     for ext in supported_formats:
# #         image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
# #     if not image_files:
# #         raise FileNotFoundError(f"No images found in the folder: {folder_path}")
    
# #     images = [cv2.imread(img_file) for img_file in image_files]
# #     return images

# # def remove_background(image, lower_color, upper_color):
# #     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# #     mask = cv2.inRange(hsv, lower_color, upper_color)
    
# #     kernel = np.ones((5, 5), np.uint8)
# #     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# #     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
# #     result = cv2.bitwise_and(image, image, mask=mask)
# #     return result

# # def focus_stack(images):
# #     stack_shape = images[0].shape[:2]
# #     focus_measure = np.zeros(stack_shape)
# #     focus_indices = np.zeros(stack_shape, dtype=int)

# #     for i, image in enumerate(images):
# #         laplacian = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
# #         mask = laplacian > focus_measure
# #         focus_measure[mask] = laplacian[mask]
# #         focus_indices[mask] = i

# #     stacked_image = np.zeros(images[0].shape, dtype=images[0].dtype)
# #     for y in range(stack_shape[0]):
# #         for x in range(stack_shape[1]):
# #             stacked_image[y, x] = images[focus_indices[y, x]][y, x]

# #     return stacked_image, focus_indices

# # def create_depth_map(focus_indices, layer_distance):
# #     depth_map = focus_indices * layer_distance
# #     return depth_map

# # def depth_map_to_point_cloud(depth_map, xy_scale=1.0, z_scale=1.0):
# #     h, w = depth_map.shape
# #     points = []

# #     for y in range(h):
# #         for x in range(w):
# #             z = depth_map[y, x] * z_scale
# #             points.append([x * xy_scale, y * xy_scale, z])

# #     return np.array(points)

# # def calculate_dimensions(points):
# #     x_min, y_min, z_min = np.min(points, axis=0)
# #     x_max, y_max, z_max = np.max(points, axis=0)
# #     length = x_max - x_min
# #     breadth = y_max - y_min
# #     height = z_max - z_min
# #     return length, breadth, height

# # class PointCloudApp(QWidget):
# #     def __init__(self):
# #         super().__init__()
# #         self.initUI()

# #     def initUI(self):
# #         self.setWindowTitle('Advanced 3D Point Cloud Reconstruction Tool')
# #         self.showMaximized()

# #         layout = QVBoxLayout()

# #         # Top layout for logo and title
# #         top_layout = QHBoxLayout()
# #         logo_label = QLabel(self)
# #         pixmap = QPixmap("logo.png")  # Add the path to your logo here
# #         logo_label.setPixmap(pixmap.scaled(150, 150, Qt.KeepAspectRatio))
# #         top_layout.addWidget(logo_label, alignment=Qt.AlignLeft)

# #         # Title section
# #         title_label = QLabel("Advanced 3D Point Cloud Reconstruction Tool", self)
# #         title_label.setFont(QFont("Arial", 24, QFont.Bold))
# #         top_layout.addWidget(title_label, alignment=Qt.AlignCenter)

# #         layout.addLayout(top_layout)

# #         # Progress bar
# #         self.progress_bar = QProgressBar(self)
# #         self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #05B8CC; }")
# #         layout.addWidget(self.progress_bar)

# #         # Dimension display label
# #         self.dimension_label = QLabel(self)
# #         self.dimension_label.setFont(QFont("Arial", 12))
# #         layout.addWidget(self.dimension_label)

# #         # Input for profile folder
# #         self.folder_input = QLineEdit(self)
# #         self.folder_input.setPlaceholderText("Select folder containing images")
# #         layout.addWidget(self.folder_input)

# #         # Browse button
# #         self.browse_button = QPushButton('Browse Folder', self)
# #         self.browse_button.setStyleSheet("background-color: #007ACC; color: white; font-weight: bold;")
# #         self.browse_button.clicked.connect(self.browse_folder)
# #         layout.addWidget(self.browse_button)

# #         # Start button
# #         self.process_button = QPushButton('Start Processing', self)
# #         self.process_button.setStyleSheet("background-color: #28A745; color: white; font-weight: bold;")
# #         self.process_button.clicked.connect(self.run_point_cloud_processing)
# #         layout.addWidget(self.process_button)

# #         # Set the main layout
# #         self.setLayout(layout)

# #     def browse_folder(self):
# #         folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
# #         if folder:
# #             self.folder_input.setText(folder)

# #     def run_point_cloud_processing(self):
# #         folder = self.folder_input.text()
# #         if not folder or not os.path.exists(folder):
# #             self.dimension_label.setText("Please select a valid folder.")
# #             return

# #         profile_folders = [folder]
        
# #         # Define HSV color range for the object
# #         lower_color = np.array([0, 0, 200])
# #         upper_color = np.array([180, 255, 255])

# #         all_point_clouds = []

# #         for folder in profile_folders:
# #             try:
# #                 images = load_images_from_folder(folder)
# #             except FileNotFoundError as e:
# #                 self.dimension_label.setText(str(e))
# #                 return
                
# #             images_cleaned = []
# #             total_images = len(images)
# #             for i, img in enumerate(images):
# #                 cleaned_img = remove_background(img, lower_color, upper_color)
# #                 images_cleaned.append(cleaned_img)

# #                 # Update progress bar and terminal output
# #                 progress = int(((i + 1) / total_images) * 100)
# #                 self.progress_bar.setValue(progress)
# #                 print(f"Processing image {i + 1}/{total_images} - {progress}% completed")

# #             stacked_image, focus_indices = focus_stack(images_cleaned)
# #             depth_map = create_depth_map(focus_indices, layer_distance=100)

# #             pixel_to_mm_scale = 0.01
# #             z_scale = 0.001

# #             point_cloud = depth_map_to_point_cloud(depth_map, xy_scale=pixel_to_mm_scale, z_scale=z_scale)
# #             all_point_clouds.append(point_cloud)

# #         merged_point_cloud = np.concatenate(all_point_clouds, axis=0)

# #         length, breadth, height = calculate_dimensions(merged_point_cloud)

# #         self.dimension_label.setText(f'Length: {length:.2f} mm, Breadth: {breadth:.2f} mm, Height: {height:.2f} mm')

# #         self.visualize_point_cloud(merged_point_cloud)

# #     def visualize_point_cloud(self, points):
# #         pcd = o3d.geometry.PointCloud()
# #         pcd.points = o3d.utility.Vector3dVector(points)

# #         coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
# #         o3d.visualization.draw_geometries([pcd, coordinate_frame])

# # if __name__ == "__main__":
# #     app = QApplication(sys.argv)
# #     ex = PointCloudApp()
# #     sys.exit(app.exec_())
# import sys
# import cv2
# import numpy as np
# import glob
# import open3d as o3d
# import os
# from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, 
#                              QPushButton, QFileDialog, QLineEdit, QProgressBar, QHBoxLayout)
# from PyQt5.QtGui import QPixmap, QFont
# from PyQt5.QtCore import Qt

# def load_images_from_folder(folder_path):
#     image_files = glob.glob(folder_path + '/*.jpg')
#     images = [cv2.imread(img_file) for img_file in image_files]
#     return images

# def remove_background(image):
#     # Convert image to LAB color space for better segmentation
#     lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
#     l_channel, a_channel, b_channel = cv2.split(lab)

#     # Thresholding on the L channel
#     _, mask = cv2.threshold(l_channel, 130, 255, cv2.THRESH_BINARY)

#     # Morphological operations to clean up the mask
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

#     result = cv2.bitwise_and(image, image, mask=mask)
#     return result, mask

# def focus_stack(images):
#     stack_shape = images[0].shape[:2]
#     focus_measure = np.zeros(stack_shape)
#     focus_indices = np.zeros(stack_shape, dtype=int)

#     for i, image in enumerate(images):
#         laplacian = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
#         mask = laplacian > focus_measure
#         focus_measure[mask] = laplacian[mask]
#         focus_indices[mask] = i

#     stacked_image = np.zeros(images[0].shape, dtype=images[0].dtype)
#     for y in range(stack_shape[0]):
#         for x in range(stack_shape[1]):
#             stacked_image[y, x] = images[focus_indices[y, x]][y, x]

#     return stacked_image, focus_indices

# def create_depth_map(focus_indices, layer_distance):
#     depth_map = focus_indices * layer_distance
#     return depth_map

# def depth_map_to_point_cloud(depth_map, xy_scale=1.0, z_scale=1.0):
#     h, w = depth_map.shape
#     points = []

#     for y in range(h):
#         for x in range(w):
#             z = depth_map[y, x] * z_scale
#             points.append([x * xy_scale, y * xy_scale, z])

#     return np.array(points)

# def calculate_dimensions(points):
#     x_min, y_min, z_min = np.min(points, axis=0)
#     x_max, y_max, z_max = np.max(points, axis=0)
#     length = x_max - x_min
#     breadth = y_max - y_min
#     height = z_max - z_min
#     return length, breadth, height

# class PointCloudApp(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.initUI()

#     def initUI(self):
#         self.setWindowTitle('Advanced 3D Point Cloud Reconstruction Tool')
#         self.showMaximized()

#         layout = QVBoxLayout()

#         # Top layout for logo and title
#         top_layout = QHBoxLayout()
#         # Logo (replace "logo.png" with your logo image path)
#         logo_label = QLabel(self)
#         pixmap = QPixmap("logo.png")
#         logo_label.setPixmap(pixmap.scaled(150, 150, Qt.KeepAspectRatio))
#         top_layout.addWidget(logo_label, alignment=Qt.AlignLeft)

#         # Title section
#         title_label = QLabel("Advanced 3D Point Cloud Reconstruction Tool", self)
#         title_label.setFont(QFont("Arial", 24, QFont.Bold))
#         top_layout.addWidget(title_label, alignment=Qt.AlignCenter)

#         layout.addLayout(top_layout)

#         # Progress bar
#         self.progress_bar = QProgressBar(self)
#         self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #05B8CC; }")
#         layout.addWidget(self.progress_bar)

#         # Dimension display label
#         self.dimension_label = QLabel(self)
#         self.dimension_label.setFont(QFont("Arial", 12))
#         layout.addWidget(self.dimension_label)

#         # Input for profile folder
#         self.folder_input = QLineEdit(self)
#         self.folder_input.setPlaceholderText("Select folder containing images")
#         layout.addWidget(self.folder_input)

#         # Browse button
#         self.browse_button = QPushButton('Browse Folder', self)
#         self.browse_button.setStyleSheet("background-color: #007ACC; color: white; font-weight: bold;")
#         self.browse_button.clicked.connect(self.browse_folder)
#         layout.addWidget(self.browse_button)

#         # Start button
#         self.process_button = QPushButton('Start Processing', self)
#         self.process_button.setStyleSheet("background-color: #28A745; color: white; font-weight: bold;")
#         self.process_button.clicked.connect(self.run_point_cloud_processing)
#         layout.addWidget(self.process_button)

#         # Set the main layout
#         self.setLayout(layout)

#     def browse_folder(self):
#         folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
#         if folder:
#             self.folder_input.setText(folder)

#     def run_point_cloud_processing(self):
#         folder = self.folder_input.text()
#         if not folder or not os.path.exists(folder):
#             self.dimension_label.setText("Please select a valid folder.")
#             return

#         profile_folders = [folder]
        
#         all_point_clouds = []

#         for folder in profile_folders:
#             images = load_images_from_folder(folder)
#             if not images:
#                 raise ValueError(f"Please provide at least two profile images in {folder}.")
                
#             images_cleaned = []
#             total_images = len(images)
#             for i, img in enumerate(images):
#                 cleaned_img, mask = remove_background(img)
#                 images_cleaned.append(cleaned_img)

#                 # Save the mask for verification
#                 mask_filename = os.path.join(folder, f"mask_{i}.png")
#                 cv2.imwrite(mask_filename, mask)

#                 # Update progress bar and terminal output
#                 progress = int(((i + 1) / total_images) * 100)
#                 self.progress_bar.setValue(progress)
#                 print(f"Processing image {i + 1}/{total_images} - {progress}% completed")

#             stacked_image, focus_indices = focus_stack(images_cleaned)
#             depth_map = create_depth_map(focus_indices, layer_distance=100)

#             pixel_to_mm_scale = 0.01
#             z_scale = 0.001

#             point_cloud = depth_map_to_point_cloud(depth_map, xy_scale=pixel_to_mm_scale, z_scale=z_scale)
#             all_point_clouds.append(point_cloud)

#         merged_point_cloud = np.concatenate(all_point_clouds, axis=0)

#         length, breadth, height = calculate_dimensions(merged_point_cloud)

#         self.dimension_label.setText(f'Length: {length:.2f} mm, Breadth: {breadth:.2f} mm, Height: {height:.2f} mm')

#         self.visualize_point_cloud(merged_point_cloud)

#     def visualize_point_cloud(self, points):
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(points)

#         coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
#         o3d.visualization.draw_geometries([pcd, coordinate_frame])

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     ex = PointCloudApp()
#     sys.exit(app.exec_())

