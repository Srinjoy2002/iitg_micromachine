import cv2
import numpy as np
import glob
import open3d as o3d
import os

def load_images_from_folder(folder_path):
    image_files = glob.glob(folder_path + '/*.jpg')
    images = [cv2.imread(img_file) for img_file in image_files]
    return images

def remove_background(image, lower_color, upper_color):
    mask = cv2.inRange(image, lower_color, upper_color)
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

def depth_map_to_point_cloud(depth_map, scale=1.0):
    h, w = depth_map.shape
    points = []

    for y in range(h):
        for x in range(w):
            z = depth_map[y, x] * scale
            points.append([x, y, z])

    return np.array(points)

def enhance_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    edges = cv2.Canny(enhanced, 100, 200)
    return edges

def detect_and_compute_keypoints(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    if descriptors is None:
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    if descriptors1 is None or descriptors2 is None:
        raise ValueError("One of the descriptor sets is None.")
    if descriptors1.shape[1] != descriptors2.shape[1]:
        raise ValueError("Descriptor dimensions do not match: {} vs {}".format(descriptors1.shape, descriptors2.shape))
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def stitch_images(base_image, image_to_stitch):
    kp1, des1 = detect_and_compute_keypoints(base_image)
    kp2, des2 = detect_and_compute_keypoints(image_to_stitch)
    
    print(f"Descriptors 1: Type: {des1.dtype}, Shape: {des1.shape}")
    if des2 is not None:
        print(f"Descriptors 2: Type: {des2.dtype}, Shape: {des2.shape}")
    else:
        print("Descriptors 2: None")
        enhanced_image = enhance_image(image_to_stitch)
        kp2, des2 = detect_and_compute_keypoints(enhanced_image)
        if des2 is None:
            raise ValueError("No descriptors found in image_to_stitch after enhancement.")
        else:
            print(f"Enhanced Descriptors 2: Type: {des2.dtype}, Shape: {des2.shape}")
    
    matches = match_features(des1, des2)
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    stitched_image = cv2.warpPerspective(image_to_stitch, H, (base_image.shape[1] + image_to_stitch.shape[1], base_image.shape[0]))
    stitched_image[0:base_image.shape[0], 0:base_image.shape[1]] = base_image
    
    return stitched_image

def visualize_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

def main():
    side_input_folder = 'cleaned_side_profile'
    top_input_folder = 'cleaned_top_view'
    
    side_images = load_images_from_folder(side_input_folder)
    top_images = load_images_from_folder(top_input_folder)

    if not side_images or not top_images:
        raise ValueError("Please provide at least two profile images.")

    lower_color = np.array([0, 0, 0])
    upper_color = np.array([180, 255, 30])

    side_images_cleaned = [remove_background(img, lower_color, upper_color) for img in side_images]
    top_images_cleaned = [remove_background(img, lower_color, upper_color) for img in top_images]

    stacked_side_image, side_focus_indices = focus_stack(side_images_cleaned)
    stacked_top_image, top_focus_indices = focus_stack(top_images_cleaned)

    stitched_image = stitch_images(stacked_side_image, stacked_top_image)

    depth_map = create_depth_map(side_focus_indices, layer_distance=1)

    point_cloud = depth_map_to_point_cloud(depth_map)

    visualize_point_cloud(point_cloud)

if __name__ == "__main__":
    main()
