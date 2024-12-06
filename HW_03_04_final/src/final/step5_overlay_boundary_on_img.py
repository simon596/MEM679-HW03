# -*-coding:utf-8 -*-

"""
#-------------------------------
# Author: Simeng Wu
# Email: sw3493@drexel.edu
#-------------------------------

#-------------------------------
"""

import cv2
import numpy as np
import glob
import os.path as osp

def overlay_bdry(original_img_path, mask_path, output_folder):
    """
    Overlay the boundary of the largest connected component (contour) from a mask onto the original image.

    Args:
        original_img_path (str): Path to the original image.
        mask_path (str): Path to the corresponding prediction mask image.
        output_folder (str): Folder where the output image with the overlayed boundary will be saved.

    Returns:
        None
    """

    # Read the original color image
    original_img = cv2.imread(original_img_path)

    # Read the mask image in grayscale mode
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Find contours in the binary mask image
    # cv2.findContours returns a list of contours and a hierarchy
    # Each contour is a list of points forming a closed polygon
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the largest contour based on its area
    largest_contour = max(contours, key=cv2.contourArea)

    # Draw the largest contour onto the original image
    # Arguments:
    #   original_img: the image onto which the contour is drawn
    #   [largest_contour]: a list of one contour (largest_contour)
    #   -1: indicates to draw this specific contour (only one in the list)
    #   (0, 255, 0): BGR color (green)
    #   thickness=2: thickness of the contour line
    cv2.drawContours(original_img, [largest_contour], -1, (0, 255, 0), 2)

    # Save the resultant image with the boundary overlay into the output folder
    cv2.imwrite(osp.join(output_folder, osp.basename(original_img_path) + ".png"), original_img)


if __name__ == '__main__':
    # Paths to the original images, their corresponding masks, and the output directory
    img_folder = r"C:\Users\sw3493\Python-Proj\Greg-acetic-06112024\raw_png"
    mask_folder = r"C:\Users\sw3493\Python-Proj\Greg-acetic-06112024\indentation_test\images\predict"
    output_folder = r"C:\Users\sw3493\Python-Proj\Greg-acetic-06112024\indentation_test\images\overlay_bdry"
    
    # Get a list of all PNG images in the specified directory
    all_img_path = glob.glob(img_folder + '/*.png')
    
    # For each image, find the corresponding mask and overlay the largest mask boundary onto the original image
    for i, img_path in enumerate(all_img_path):
        mask_path = osp.join(mask_folder, osp.basename(img_path))
        overlay_bdry(img_path, mask_path, output_folder)
