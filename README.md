**Overview**

This project aims to create accurate 3D point clouds from 2D images of tool profiles using advanced computer vision techniques. It combines image processing, feature matching, focus stacking, and 3D modeling to generate detailed 3D representations of tools.

**Features**

Background Removal: Uses color thresholding and contour detection to isolate the tool from its background.

Feature Matching: Employs the ORB (Oriented FAST and Rotated BRIEF) algorithm for keypoint detection and matching.

Focus Stacking: Combines multiple images to enhance depth information using the variance of Laplacian method.

3D Modeling: Generates depth maps from focus-stacked images and converts them into 3D point clouds.

Visualization: Renders 3D models using Open3D for interactive visualization.

**Requirements**

1.Python 3.7+

2.OpenCV 4.5+

3.NumPy

4.Matplotlib

5.Open3D

6.glob

7.mayavi (optional for additional 3D visualization)
