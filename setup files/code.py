import sys
import cv2
import numpy as np
import glob
import open3d as o3d
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel

# Define pixel_to_mm_scale based on the effective pixel size
pixel_to_mm_scale = 0.0029  # Adjusted based on effective pixel size

def load_images_from_folder(folder_path):
    image_files = glob.glob(folder_path + '/*.jpg')
    images = [cv2.imread(img_file) for img_file in image_files]
    return images

def convert_to_binary(image, threshold=100):  # Lowered threshold value
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary

def largest_continuous_region(binary_image):
    num_labels, labels_im = cv2.connectedComponents(binary_image)
    if num_labels > 1:  # Ensure there are connected components
        max_label = 1 + np.argmax(np.bincount(labels_im.flat)[1:])
        largest_region = np.zeros_like(binary_image)
        largest_region[labels_im == max_label] = 255
        return largest_region
    else:
        print("No connected components found.")
        return binary_image  # Return the original binary image if no components found

def focus_stack(images):
    stack_shape = images[0].shape[:2]
    focus_measure = np.zeros(stack_shape)
    focus_indices = np.zeros(stack_shape, dtype=int)

    for i, image in enumerate(images):
        largest_region = largest_continuous_region(image)
        laplacian = cv2.Laplacian(largest_region, cv2.CV_64F)
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
            if z > 0:  # Only include points where there is depth information
                points.append([x * xy_scale, y * xy_scale, z])

    return np.array(points)

def calculate_dimensions(points):
    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)
    length = (x_max - x_min) * pixel_to_mm_scale  # Convert to mm
    breadth = (y_max - y_min) * pixel_to_mm_scale  # Convert to mm
    height = (z_max - z_min) * z_scale  # Already in mm
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
        profile_folders = ['side_1', 'side_2', 'side_3', 'side_4']
        all_point_clouds = []

        for folder in profile_folders:
            images = load_images_from_folder(folder)
            if not images:
                raise ValueError(f"Please provide at least two profile images in {folder}.")

            images_binary = [convert_to_binary(img) for img in images]

            for i, binary_image in enumerate(images_binary):
                cv2.imwrite(f'{folder}_binary_{i}.png', binary_image)

            stacked_image, focus_indices = focus_stack(images_binary)
            depth_map = create_depth_map(focus_indices, layer_distance=100)  # 100 microns

            z_scale = 0.001  # Adjust z-axis scaling factor

            point_cloud = depth_map_to_point_cloud(depth_map, xy_scale=pixel_to_mm_scale, z_scale=z_scale)
            all_point_clouds.append(point_cloud)

        merged_point_cloud = np.concatenate(all_point_clouds, axis=0)

        length, breadth, height = calculate_dimensions(merged_point_cloud)

        self.dimension_label.setText(f'Length: {length:.2f} mm, Breadth: {breadth:.2f} mm, Height: {height:.2f} mm')

        self.visualize_point_cloud(merged_point_cloud)

    def visualize_point_cloud(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, coordinate_frame])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = PointCloudApp()
    sys.exit(app.exec_())
