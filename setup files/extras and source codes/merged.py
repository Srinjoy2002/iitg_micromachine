import cv2
import numpy as np
import glob
import open3d as o3d

# Function to load images from a specified folder
def load_images_from_folder(folder_path):
    image_files = glob.glob(folder_path + '/*.jpg')  # Adjust the file extension as needed
    images = [cv2.imread(img_file) for img_file in image_files]
    return images

# Load side and top profile images
side_images = load_images_from_folder(r'D:\iitg\tut\side profile of tool')
top_images = load_images_from_folder(r'D:\iitg\tut\top view')

if not side_images or not top_images:
    raise ValueError("No images found in one of the specified directories.")

# Function to compute the variance of the Laplacian (a measure of focus)
def variance_of_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F)

# Function to focus stack images
def focus_stack(images):
    stack_shape = images[0].shape[:2]
    focus_measure = np.zeros(stack_shape)
    focus_indices = np.zeros(stack_shape, dtype=int)

    for i, image in enumerate(images):
        laplacian = variance_of_laplacian(image)
        mask = laplacian > focus_measure
        focus_measure[mask] = laplacian[mask]
        focus_indices[mask] = i

    stacked_image = np.zeros_like(images[0])
    for i, image in enumerate(images):
        stacked_image[focus_indices == i] = image[focus_indices == i]

    return stacked_image, focus_indices

# Process images to get focus-stacked images and depth maps
stacked_side_image, side_focus_indices = focus_stack(side_images)
stacked_top_image, top_focus_indices = focus_stack(top_images)

cv2.imwrite('stacked_side_image.jpg', stacked_side_image)
cv2.imwrite('stacked_top_image.jpg', stacked_top_image)

# Function to create depth map from focus indices
def create_depth_map(focus_indices, layer_distance):
    depth_map = focus_indices * layer_distance
    return depth_map

# Set layer distances
side_layer_distance = 100  # microns
top_layer_distance = 10    # microns

side_depth_map = create_depth_map(side_focus_indices, side_layer_distance)
top_depth_map = create_depth_map(top_focus_indices, top_layer_distance)

cv2.imwrite('side_depth_map.jpg', side_depth_map)
cv2.imwrite('top_depth_map.jpg', top_depth_map)

# Function to convert depth map to point cloud with color
def depth_map_to_point_cloud_with_color(depth_map, image, scale=1.0):
    h, w = depth_map.shape
    points = []
    colors = []

    for y in range(h):
        for x in range(w):
            z = depth_map[y, x] * scale
            points.append([x, y, z])
            colors.append(image[y, x] / 255.0)  # Normalize color to [0, 1]

    return np.array(points), np.array(colors)

# Generate point clouds from depth maps with color information
side_point_cloud, side_colors = depth_map_to_point_cloud_with_color(side_depth_map, stacked_side_image)
top_point_cloud, top_colors = depth_map_to_point_cloud_with_color(top_depth_map, stacked_top_image)

# Save point clouds
np.savetxt('side_point_cloud.xyz', side_point_cloud, fmt='%.2f')
np.savetxt('top_point_cloud.xyz', top_point_cloud, fmt='%.2f')

# Function to refine point cloud
def refine_point_cloud(points, colors, voxel_size=0.05):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Downsample
    pcd = pcd.voxel_down_sample(voxel_size)
    
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Remove outliers
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    
    return pcd

# Load and refine the point clouds
refined_side_pcd = refine_point_cloud(side_point_cloud, side_colors)
refined_top_pcd = refine_point_cloud(top_point_cloud, top_colors)

# Align and merge point clouds
def align_and_merge_point_clouds(pcd1, pcd2):
    threshold = 0.02  # Distance threshold for ICP
    trans_init = np.eye(4)  # Initial transformation

    # Coarse alignment using Fast Global Registration
    fgr = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        pcd2, pcd1,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=threshold))

    # Fine alignment using ICP
    icp = o3d.pipelines.registration.registration_icp(
        pcd2, pcd1, threshold, fgr.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    pcd2.transform(icp.transformation)
    merged_pcd = pcd1 + pcd2
    merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.02)

    return merged_pcd

merged_pcd = align_and_merge_point_clouds(refined_side_pcd, refined_top_pcd)

# Visualize the merged point cloud
def visualize_point_cloud(pcd):
    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.8,
                                      front=[0.5, 0.5, -1],
                                      lookat=[0, 0, 0],
                                      up=[0, -1, 0])

visualize_point_cloud(merged_pcd)

# Save the merged point cloud
o3d.io.write_point_cloud('merged_point_cloud.ply', merged_pcd)
