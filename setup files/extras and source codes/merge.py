import cv2
import numpy as np
import glob
import os

def load_images_from_folder(folder_path):
    image_files = glob.glob(folder_path + '/*.jpg')  
    images = [cv2.imread(img_file, cv2.IMREAD_GRAYSCALE) for img_file in image_files]
    return images

def focus_stack(images):
    stack_shape = images[0].shape
    focus_measure = np.zeros(stack_shape)
    focus_indices = np.zeros(stack_shape, dtype=int)

    for i, image in enumerate(images):
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        mask = laplacian > focus_measure
        focus_measure[mask] = laplacian[mask]
        focus_indices[mask] = i

    stacked_image = np.zeros(stack_shape, dtype=images[0].dtype)
    for i, image in enumerate(images):
        stacked_image[focus_indices == i] = image[focus_indices == i]

    return stacked_image, focus_indices

def create_depth_map(focus_indices, layer_distance):
    depth_map = focus_indices * layer_distance
    return depth_map

def depth_map_to_point_cloud(depth_map, scale=1.0):
    h, w = depth_map.shape
    points = []

    for y in range(h):
        for x in range(w):
            z = depth_map[y, x] * scale
            points.append([x, y, z])

    return np.array(points)

def generate_point_cloud(side_folder, top_folder, target_size, layer_distance):
    
    side_images = load_images_from_folder(side_folder)
    top_images = load_images_from_folder(top_folder)

    
    stacked_side_image, side_focus_indices = focus_stack(side_images)

    
    stacked_top_image, top_focus_indices = focus_stack(top_images)

   
    side_depth_map = create_depth_map(side_focus_indices, layer_distance)
    top_depth_map = create_depth_map(top_focus_indices, layer_distance)

    print("Original side depth map size:", side_depth_map.shape)
    print("Original top depth map size:", top_depth_map.shape)
    print("Target size:", target_size)

    
    side_depth_map_resized = cv2.resize(side_depth_map, (target_size[1], target_size[0]))
    top_depth_map_resized = cv2.resize(top_depth_map, (target_size[1], target_size[0]))

    print("Resized side depth map size:", side_depth_map_resized.shape)
    print("Resized top depth map size:", top_depth_map_resized.shape)

    
    combined_depth_map = np.maximum(side_depth_map_resized, top_depth_map_resized)

    
    point_cloud = depth_map_to_point_cloud(combined_depth_map)

    return point_cloud, side_depth_map_resized, top_depth_map_resized, stacked_side_image, stacked_top_image


side_folder = r'sfm side'
top_folder = r'sfm top'


target_size = (753, 513)


layer_distance = 10


point_cloud, side_depth_map, top_depth_map, stacked_side_image, stacked_top_image = generate_point_cloud(side_folder, top_folder, target_size, layer_distance)


np.savetxt('point_cloud.xyz', point_cloud, fmt='%.2f')


cv2.imwrite('side_depth_map.jpg', side_depth_map)
cv2.imwrite('top_depth_map.jpg', top_depth_map)


cv2.imwrite('stacked_side_image.jpg', stacked_side_image)
cv2.imwrite('stacked_top_image.jpg', stacked_top_image)
