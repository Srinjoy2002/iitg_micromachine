import cv2
import numpy as np
import glob
import open3d as o3d

# Load images
image_files = glob.glob(r'D:\iitg\tut\side profile of tool\*.jpg')  # Adjust the path and file extension as needed
images = [cv2.imread(img_file, cv2.IMREAD_GRAYSCALE) for img_file in image_files]

if not images:
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

stacked_image, focus_indices = focus_stack(images)
cv2.imwrite('stacked_image2.jpg', stacked_image)

def create_depth_map(focus_indices, layer_distance):
    depth_map = focus_indices * layer_distance
    return depth_map

layer_distance = 100  # microns
depth_map = create_depth_map(focus_indices, layer_distance)
cv2.imwrite('depth_map2.jpg', depth_map)

def depth_map_to_point_cloud(depth_map, scale=1.0):
    h, w = depth_map.shape
    points = []

    for y in range(h):
        for x in range(w):
            z = depth_map[y, x] * scale
            points.append([x, y, z])

    return np.array(points)

# Generate point cloud and refine it
point_cloud = depth_map_to_point_cloud(depth_map)
np.savetxt('point_cloud2.xyz', point_cloud, fmt='%.2f')

# Refine point cloud
def refine_point_cloud(points, voxel_size=0.05):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Downsample
    pcd = pcd.voxel_down_sample(voxel_size)
    
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Remove outliers
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    return pcd

# Load and refine the point cloud
points = np.loadtxt('point_cloud2.xyz')
refined_pcd = refine_point_cloud(points)

# Visualize the refined point cloud
def visualize_point_cloud(pcd):
    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

visualize_point_cloud(refined_pcd)

# Save the refined point cloud
o3d.io.write_point_cloud('refined_point_cloud1.ply', refined_pcd)
