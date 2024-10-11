import sys
import cv2
import numpy as np
import glob
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision import transforms as T
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QProgressBar

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
        
        # Check if any masks are available
        if 'masks' not in prediction or len(prediction['masks']) == 0:
            raise ValueError("No valid masks detected")
        
        # Get the mask with the highest score
        mask = prediction['masks'][0, 0].mul(255).byte().cpu().numpy()
        return mask

def load_images_from_folder(folder_path):
    image_files = glob.glob(folder_path + '/*.jpg')
    if not image_files:
        raise FileNotFoundError(f"No images found in the folder: {folder_path}")
    images = [cv2.imread(img_file) for img_file in image_files]
    return images

def mask_image_with_rcnn(image):
    try:
        mask = get_object_mask(image)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        return masked_image
    except ValueError as e:
        print(f"Skipping image due to error: {e}")
        return image

# Focus stacking algorithm
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

# Create depth map from focus indices
def create_depth_map(focus_indices, layer_distance):
    depth_map = focus_indices * layer_distance
    return depth_map

# Generate point cloud with colors
def depth_map_to_point_cloud(depth_map, images, xy_scale=1.0, z_scale=1.0):
    h, w = depth_map.shape
    points = []
    colors = []

    max_index = len(images) - 1  # Ensure the index does not exceed the available images

    for y in range(h):
        for x in range(w):
            z = depth_map[y, x] * z_scale
            points.append([x * xy_scale, y * xy_scale, z])
            
            # Ensure that the index is valid (within bounds)
            image_index = min(depth_map[y, x], max_index)
            
            # Extract color from corresponding pixel
            colors.append(images[image_index][y, x] / 255.0)  # Normalize the color to [0, 1]

    return np.array(points), np.array(colors)

# Calculate dimensions (length, breadth, height) of the object
def calculate_dimensions(points):
    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)
    length = x_max - x_min
    breadth = y_max - y_min
    height = z_max - z_min
    return length, breadth, height

# RMS Error calculation
def calculate_rms_error(predicted, actual):
    return np.sqrt(np.mean((np.array(predicted) - np.array(actual))**2))

# Simulated "ground truth" or actual measurements for comparison
def get_simulated_actual_measurements():
    # These are simulated actual measurements in mm (replace with real data later)
    return [100.0, 50.0, 30.0]  # Length, Breadth, Height in mm

# Main app to handle point cloud processing
class PointCloudApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Point Cloud and Dimensions')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.dimension_label = QLabel(self)
        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.dimension_label)
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)
        self.show()

        self.run_point_cloud_processing()

    def run_point_cloud_processing(self):
        folder = 'D:/iitg/iitg_micromachine/setup files/cleaned_images'  # Update this to the folder containing your images

        try:
            images = load_images_from_folder(folder)
        except FileNotFoundError as e:
            self.dimension_label.setText(f"Error: {e}")
            return

        # Progress bar setup
        self.progress_bar.setMaximum(len(images))
        
        # Mask R-CNN for background removal
        images_cleaned = []
        for i, img in enumerate(images):
            masked_img = mask_image_with_rcnn(img)
            images_cleaned.append(masked_img)
            self.progress_bar.setValue(i + 1)

        if len(images_cleaned) == 0:
            self.dimension_label.setText("Error: No valid images after Mask R-CNN processing.")
            return
        
        stacked_image, focus_indices = focus_stack(images_cleaned)
        depth_map = create_depth_map(focus_indices, layer_distance=100)  # Adjust layer distance as needed

        # Scale adjustments
        pixel_to_mm_scale = 0.01  # Adjust according to your pixel size
        z_scale = 0.001

        point_cloud, colors = depth_map_to_point_cloud(depth_map, images_cleaned, xy_scale=pixel_to_mm_scale, z_scale=z_scale)
        length, breadth, height = calculate_dimensions(point_cloud)

        # Simulated actual measurements (replace with real data)
        actual_measurements = get_simulated_actual_measurements()

        # Calculating RMS error
        predicted_measurements = [length, breadth, height]
        rms_error = calculate_rms_error(predicted_measurements, actual_measurements)

        self.dimension_label.setText(f'Length: {length:.2f} mm, Breadth: {breadth:.2f} mm, Height: {height:.2f} mm\n'
                                     f'RMS Error: {rms_error:.2f} mm')

        self.visualize_point_cloud(point_cloud, colors)

    def visualize_point_cloud(self, points, colors):
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Adding coordinate frame for better orientation
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

        # Interactive visualization of the point cloud
        o3d.visualization.draw_geometries([pcd, coordinate_frame], 
                                          zoom=0.5, 
                                          front=[0.0, 0.0, -1.0],
                                          lookat=[0.0, 0.0, 0.0],
                                          up=[0.0, -1.0, 0.0])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = PointCloudApp()
    sys.exit(app.exec_())
