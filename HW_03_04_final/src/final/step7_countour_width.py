# -*- coding: utf-8 -*-

"""
#-------------------------------
# Author: Simeng Wu
# Email: sw3493@drexel.edu
#-------------------------------
# Creation Date: Feb 28th, 2024, EST.
#-------------------------------
"""

import cv2
import numpy as np
import glob
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from typing import List
import csv

def find_width(mask_path):
    """
    Given a binary mask image, this function identifies the largest object (by contour area) 
    and computes the average thickness (width) and height of that object, focusing on the 
    main body of the shape. The mask is assumed to represent a "strip-like" structure, and 
    the function attempts to extract the characteristic width and height by looking at the 
    contour coordinates.

    Steps:
        1. Read the mask image in grayscale.
        2. Find contours in the mask. Each contour represents a closed shape.
        3. Identify the largest contour based on its area.
        4. Extract the x and y coordinates of this contour.
        5. Compute the bounding values (min_x, max_x, min_y, max_y).
        6. Find the average coordinates along the contour to estimate where to separate 
           left/right and top/bottom boundaries.
        7. Filter contour points based on their positions to identify points lying on 
           the left, right, top, and bottom boundaries, excluding areas near the extremes 
           (using a small percentage cutoff).
        8. Compute the mean positions of these filtered points on each boundary.
        9. The width and height of the strip are computed as the difference between the 
           mean positions of the right and left boundary points (for width), and the 
           difference between top and bottom boundary points (for height).

    Args:
        mask_path (str): The path to the binary mask image.

    Returns:
        (float, float): A tuple (width_strip, height_strip) representing the computed width 
                        and height of the largest contour's main body.
    """
    # Load the binary mask image in grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Find contours in the mask image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Select the largest contour based on area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Contours are returned as [[[x,y]],[[x,y]],...], reshape to [[x,y],[x,y],...]
    contour_pts = largest_contour[:, 0, :]
    # Compute bounding values and size
    min_y = np.min(contour_pts[:, 1])
    max_y = np.max(contour_pts[:, 1])
    h = max_y - min_y
    min_x = np.min(contour_pts[:, 0])
    max_x = np.max(contour_pts[:, 0])
    w = max_x - min_x
    
    # Compute average coordinates to split the contour into left/right and top/bottom sets
    x_avg = np.mean(contour_pts[:, 0])
    y_avg = np.mean(contour_pts[:, 1])

    # Filter points to find those on the left and right boundaries, excluding top and bottom parts 
    # by applying a margin (0.03 * h) from the top and bottom of the shape.
    left_boundary_pts = contour_pts[
        (contour_pts[:, 0] < x_avg) & 
        (contour_pts[:, 1] > min_y + 0.03 * h) & 
        (contour_pts[:, 1] < max_y - 0.03 * h)
    ]
    right_boundary_pts = contour_pts[
        (contour_pts[:, 0] > x_avg) & 
        (contour_pts[:, 1] > min_y + 0.03 * h) & 
        (contour_pts[:, 1] < max_y - 0.03 * h)
    ]

    # Similarly, filter points to find those on the top and bottom boundaries, excluding 
    # areas close to the left and right edges (0.03 * w)
    bot_boundary_pts = contour_pts[
        (contour_pts[:, 1] < y_avg) & 
        (contour_pts[:, 0] > min_x + 0.03 * w) & 
        (contour_pts[:, 0] < max_x - 0.03 * w)
    ]
    top_boundary_pts = contour_pts[
        (contour_pts[:, 1] > y_avg) & 
        (contour_pts[:, 0] > min_x + 0.03 * w) & 
        (contour_pts[:, 0] < max_x - 0.03 * w)
    ]

    # Compute mean x-coordinates for left and right boundaries
    x_mean_left_boundary = np.mean(left_boundary_pts[:, 0])
    x_mean_right_boundary = np.mean(right_boundary_pts[:, 0])

    # Compute mean y-coordinates for bottom and top boundaries
    y_mean_bot_boundary = np.mean(bot_boundary_pts[:, 1])
    y_mean_top_boundary = np.mean(top_boundary_pts[:, 1])
    
    # Compute the width and height of the strip as the difference between boundary averages
    width_strip = x_mean_right_boundary - x_mean_left_boundary
    height_strip = y_mean_top_boundary - y_mean_bot_boundary
    return width_strip, height_strip


def img_folder_to_str_list(folder_path: Path) -> List[str]:
    """
    Given a folder path, return a sorted list of file paths (as strings) to PNG images in that folder.

    Args:
        folder_path (Path): The path to a folder containing PNG images.

    Returns:
        List[str]: A sorted list of image file paths as strings.
    """
    # Find all PNG files in the given folder
    name_list = glob.glob(str(folder_path) + '/*.png')
    name_list.sort()
    name_list_str = [str(name) for name in name_list]
    return name_list_str


def give_widths_list(image_folder: List[str]) -> List[tuple]:
    """
    For each image path in the input list, compute the width and height using 'find_width' and 
    return the results as a list of tuples.

    Args:
        image_folder (List[str]): A list of image file paths.

    Returns:
        List[tuple]: A list where each element is (width, height) for the corresponding image.
    """
    widthNheight_list = []
    for idx, img_path in enumerate(image_folder):
        widthNheight = find_width(img_path)
        widthNheight_list.append(widthNheight)
    return widthNheight_list


if __name__ == '__main__':
    # Define the directory containing predicted mask images
    raw_masks_path = Path(".\\images\\predict")
    # Obtain a sorted list of mask image file paths
    figures_list = img_folder_to_str_list(raw_masks_path)
    # Compute the width and height for each image
    widthNheight_list = give_widths_list(figures_list)
    
    # Write the computed widths (and optionally heights) to a CSV file
    csv_file_path = r".\results"
    csv_file_path = osp.join(csv_file_path, 'DNgel01_widths_UNet.csv')
    
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        # Only the width is written to each line in this case. 
        # If desired, both width and height could be written.
        for (width, height) in widthNheight_list:
            # Uncomment the following to write both width and height:
            # writer.writerow([width, height])
            writer.writerow([width])
    
    print(f"File 'DNgel01_widths_UNet.csv' written successfully and saved to {csv_file_path}")
