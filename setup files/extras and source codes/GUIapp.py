import sys
import cv2
import numpy as np
import glob
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision import transforms as T
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel

# Load Mask R-CNN model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
model.to(device)

# Preprocessing function for images
def preprocess_image(image):
    transform = T.Compose([T.ToTensor()])
    return transform(image).unsqueeze(0).to(device)

# Function to get mask from Mask R-CNN
def get_object_mask(image):
    with torch.no_grad():
        # Preprocess the image for R-CNN
        image_tensor = preprocess_image(image)
        
        # Get prediction from the model
        prediction = model(image_tensor)[0]
        
        # Get the mask with the highest score (assuming the object is the primary element)
        mask = prediction['masks'][0, 0].mul(255).byte().cpu().numpy()
        return mask

def load_images_from_folder(folder_path):
    image_files = glob.glob(folder_path + '/*.jpg')
    images = [cv2.imread(img_file) for img_file in image_files]
    return images

def mask_image_with_rcnn(image):
    # Get the mask from the R-CNN model
    mask = get_object_mask(image)
    
    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

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
        folder = 'data\niddles'  # Update this to the folder containing your single profile images

        images = load_images_from_folder(folder)
        if not images:
            raise ValueError(f"Please provide images in {folder}.")
            
        # Use Mask R-CNN for background removal
        images_cleaned = [mask_image_with_rcnn(img) for img in images]

        if len(images_cleaned) == 0:
            raise ValueError(f"No valid images found in {folder}.")
        
        stacked_image, focus_indices = focus_stack(images_cleaned)
        depth_map = create_depth_map(focus_indices, layer_distance=100)  # Adjust layer distance as needed

        pixel_to_mm_scale = 0.01  # Adjust the scaling factor according to your pixel size
        z_scale = 0.001

        point_cloud = depth_map_to_point_cloud(depth_map, xy_scale=pixel_to_mm_scale, z_scale=z_scale)
        length, breadth, height = calculate_dimensions(point_cloud)

        self.dimension_label.setText(f'Length: {length:.2f} mm, Breadth: {breadth:.2f} mm, Height: {height:.2f} mm')

        self.visualize_point_cloud(point_cloud)

    def visualize_point_cloud(self, points):
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, coordinate_frame])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = PointCloudApp()
    sys.exit(app.exec_())
