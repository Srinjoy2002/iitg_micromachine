import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QVBoxLayout, QWidget
import cv2
import numpy as np
import open3d as o3d
import glob

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("3D Image Processor")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()

        self.load_side_images_btn = QPushButton("Load Side Profile Images")
        self.load_side_images_btn.clicked.connect(self.load_side_images)
        self.layout.addWidget(self.load_side_images_btn)

        self.load_top_images_btn = QPushButton("Load Top Profile Images")
        self.load_top_images_btn.clicked.connect(self.load_top_images)
        self.layout.addWidget(self.load_top_images_btn)

        self.process_btn = QPushButton("Process Images")
        self.process_btn.clicked.connect(self.process_images)
        self.layout.addWidget(self.process_btn)

        self.export_btn = QPushButton("Export Point Cloud")
        self.export_btn.clicked.connect(self.export_point_cloud)
        self.layout.addWidget(self.export_btn)

        self.status_label = QLabel("Status: Ready")
        self.layout.addWidget(self.status_label)

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        self.side_images = []
        self.top_images = []

    def load_side_images(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        files, _ = QFileDialog.getOpenFileNames(self, "Load Side Profile Images", "", "Images (*.png *.xpm *.jpg)", options=options)
        if files:
            self.side_images = [cv2.imread(file) for file in files]
            self.status_label.setText(f"Loaded {len(files)} side profile images.")

    def load_top_images(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        files, _ = QFileDialog.getOpenFileNames(self, "Load Top Profile Images", "", "Images (*.png *.xpm *.jpg)", options=options)
        if files:
            self.top_images = [cv2.imread(file) for file in files]
            self.status_label.setText(f"Loaded {len(files)} top profile images.")

    def process_images(self):
        if not self.side_images or not self.top_images:
            self.status_label.setText("Please load both side and top profile images.")
            return

        self.stacked_side_image, self.side_focus_indices = self.focus_stack(self.side_images)
        self.stacked_top_image, self.top_focus_indices = self.focus_stack(self.top_images)

        self.side_depth_map = self.create_depth_map(self.side_focus_indices, 100)  # microns
        self.top_depth_map = self.create_depth_map(self.top_focus_indices, 10)    # microns

        self.side_point_cloud, self.side_colors = self.depth_map_to_point_cloud_with_color(self.side_depth_map, self.stacked_side_image)
        self.top_point_cloud, self.top_colors = self.depth_map_to_point_cloud_with_color(self.top_depth_map, self.stacked_top_image)

        self.refined_side_pcd = self.refine_point_cloud(self.side_point_cloud, self.side_colors)
        self.refined_top_pcd = self.refine_point_cloud(self.top_point_cloud, self.top_colors)

        self.merged_pcd = self.align_and_merge_point_clouds(self.refined_side_pcd, self.refined_top_pcd)

        self.status_label.setText("Images processed successfully.")

    def export_point_cloud(self):
        if not hasattr(self, 'merged_pcd'):
            self.status_label.setText("Please process images first.")
            return

        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        save_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Point Cloud", options=options)
        if save_dir:
            o3d.io.write_point_cloud(f"{save_dir}/merged_point_cloud.ply", self.merged_pcd)
            o3d.io.write_point_cloud(f"{save_dir}/merged_point_cloud.obj", self.merged_pcd)
            o3d.io.write_point_cloud(f"{save_dir}/merged_point_cloud.stl", self.merged_pcd)
            self.status_label.setText("Point cloud exported successfully.")

    def focus_stack(self, images):
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
        for i, image in enumerate(images):
            stacked_image[focus_indices == i] = image[focus_indices == i]

        return stacked_image, focus_indices

    def create_depth_map(self, focus_indices, layer_distance):
        depth_map = focus_indices * layer_distance
        return depth_map

    def depth_map_to_point_cloud_with_color(self, depth_map, image, scale=1.0):
        h, w = depth_map.shape
        points = []
        colors = []

        for y in range(h):
            for x in range(w):
                z = depth_map[y, x] * scale
                points.append([x, y, z])
                colors.append(image[y, x] / 255.0)  # Normalize color to [0, 1]

        return np.array(points), np.array(colors)

    def refine_point_cloud(self, points, colors, voxel_size=0.05):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd = pcd.voxel_down_sample(voxel_size)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.select_by_index(ind)

        return pcd

    def align_and_merge_point_clouds(self, pcd1, pcd2):
        threshold = 0.02  # Distance threshold for ICP
        trans_init = np.eye(4)  # Initial transformation

        fgr = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
            pcd2, pcd1,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=threshold))

        icp = o3d.pipelines.registration.registration_icp(
            pcd2, pcd1, threshold, fgr.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())

        pcd2.transform(icp.transformation)
        merged_pcd = pcd1 + pcd2
        merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.02)

        return merged_pcd

# Run the application
app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
