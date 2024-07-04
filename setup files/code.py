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
        profile_folders = ['side profile cap','top profile cap']
        
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
