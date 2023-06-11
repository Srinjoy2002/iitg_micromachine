import cv2
import numpy as np
import open3d as o3d

def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    return image

def image_to_point_cloud(image, scale=1.0, step=10):
    h, w = image.shape
    points = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            if image[y, x] != 0:  # Exclude background points
                z = image[y, x] * scale
                points.append([x, y, z])
    return np.array(points)

def create_deformed_cone(cone_height, base_radius, tip_radius, height_ratio, tip_height, noise_level=5.0, cut_angle=np.pi/4):
    points = []

    # Create deformed cylindrical base
    cylinder_height = cone_height * height_ratio
    for z in np.linspace(0, cylinder_height, int(cylinder_height)):
        for angle in np.linspace(0, 2 * np.pi, 360):
            if angle < 2 * np.pi - cut_angle:  # Create a cut in the cone
                x = base_radius * np.cos(angle) + np.random.uniform(-noise_level, noise_level)
                y = base_radius * np.sin(angle) + np.random.uniform(-noise_level, noise_level)
                points.append([x, y, z])

    # Create deformed conical top
    for z in np.linspace(0, cone_height - tip_height, int(cone_height - tip_height)):
        for angle in np.linspace(0, 2 * np.pi, 360):
            if angle < 2 * np.pi - cut_angle:  # Create a cut in the cone
                radius = base_radius * (1 - z / cone_height) + np.random.uniform(-noise_level, noise_level)
                x = radius * np.cos(angle) + np.random.uniform(-noise_level, noise_level)
                y = radius * np.sin(angle) + np.random.uniform(-noise_level, noise_level)
                points.append([x, y, z + cylinder_height])

    # Create deformed cylindrical tip
    for z in np.linspace(0, tip_height, int(tip_height)):
        for angle in np.linspace(0, 2 * np.pi, 360):
            if angle < 2 * np.pi - cut_angle:  # Create a cut in the cone
                x = tip_radius * np.cos(angle) + np.random.uniform(-noise_level, noise_level)
                y = tip_radius * np.sin(angle) + np.random.uniform(-noise_level, noise_level)
                points.append([x, y, z + cylinder_height + (cone_height - tip_height)])

    return np.array(points)

def adjust_points_to_fit_structure(points, structure_points):
    max_z_structure = np.max(structure_points[:, 2])
    max_z_points = np.max(points[:, 2])

    scale_factor = max_z_structure / max_z_points
    adjusted_points = points * scale_factor

    return adjusted_points

def visualize_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

# File path
side_image_path = 'sidebg.jpg'

# Load image
side_image = load_image(side_image_path)

# Generate point cloud from image
side_point_cloud = image_to_point_cloud(side_image, scale=1.0, step=10)

# Create deformed artificial cone structure with noise
cone_height = 500  # Height of the cone
base_radius = 200  # Radius of the cylindrical base
tip_radius = 50  # Radius of the cylindrical tip
height_ratio = 0.2  # Ratio of base height to cone height
tip_height = 100  # Height of the cylindrical tip
noise_level = 10.0  # Higher noise level for more deformation
cut_angle = np.pi / 4  # Angle to cut out a section of the cone

deformed_cone_points = create_deformed_cone(cone_height, base_radius, tip_radius, height_ratio, tip_height, noise_level, cut_angle)

# Adjust generated point cloud to fit inside the deformed artificial structure
adjusted_side_point_cloud = adjust_points_to_fit_structure(side_point_cloud, deformed_cone_points)

# Combine the point clouds
combined_point_cloud = np.vstack((deformed_cone_points, adjusted_side_point_cloud))
np.savetxt('combined_point_cloud.xyz', combined_point_cloud, fmt='%.2f')

# Visualize combined point cloud in Open3D viewer
visualize_point_cloud(combined_point_cloud)
