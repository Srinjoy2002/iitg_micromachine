import cv2
import numpy as np
import glob
import open3d as o3d

def load_images_from_folder(folder_path, target_size):
    image_files = glob.glob(folder_path + '/*.jpg')  # Adjust the file extension as needed
    images = [cv2.imread(img_file, cv2.IMREAD_GRAYSCALE) for img_file in image_files]
    resized_images = [cv2.resize(img, target_size) for img in images]
    return resized_images

# Load and resize images
path_top = r'D:/iitg/tut/top view resized'
top_images = load_images_from_folder(path_top, (513, 753))  # Adjust the target size to a common shape

if not top_images:
    raise ValueError("No images found in the specified directory.")

# Function to compute the variance of the Laplacian (a measure of focus)
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F)

def focus_stack(images):
    stack_shape = images[0].shape
    focus_measure = np.zeros(stack_shape)
    focus_indices = np.zeros(stack_shape, dtype=int)

    for i, image in enumerate(images):
        laplacian = variance_of_laplacian(image)
        mask = laplacian > focus_measure
        focus_measure[mask] = laplacian[mask]
        focus_indices[mask] = i

    stacked_image = np.zeros(stack_shape, dtype=images[0].dtype)
    for i, image in enumerate(images):
        stacked_image[focus_indices == i] = image[focus_indices == i]

    return stacked_image, focus_indices

# Process top profile images to get focus-stacked images and depth map
stacked_top_image, top_focus_indices = focus_stack(top_images)
cv2.imwrite('stacked_top_image.jpg', stacked_top_image)

def create_depth_map(focus_indices, layer_distance):
    depth_map = focus_indices * layer_distance
    return depth_map

layer_distance = 10  # microns
top_depth_map = create_depth_map(top_focus_indices, layer_distance)
cv2.imwrite('top_depth_map.jpg', top_depth_map)

def depth_map_to_point_cloud(depth_map, scale=1.0):
    h, w = depth_map.shape
    points = []

    for y in range(h):
        for x in range(w):
            z = depth_map[y, x] * scale
            points.append([x, y, z])

    return np.array(points)

# Generate point cloud from depth map
top_point_cloud = depth_map_to_point_cloud(top_depth_map)
np.savetxt('top_point_cloud.xyz', top_point_cloud, fmt='%.2f')

# Function to visualize point cloud using Open3D
def visualize_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

# Load the point cloud from the saved file and visualize it
points = np.loadtxt('top_point_cloud.xyz')
visualize_point_cloud(points)
