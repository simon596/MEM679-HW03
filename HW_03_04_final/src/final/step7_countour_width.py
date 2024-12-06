#%%
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
    # Load the prediction mask (ensure it's in grayscale)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Find contours from the binary image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Identify the largest contour based on the area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Reshape from [[[x,y]],[[x,y]],...] to [[x,y],[x,y],...]
    contour_pts = largest_contour[:,0,:]
    print(type(contour_pts))
    min_y = np.min(contour_pts[:,1])
    max_y = np.max(contour_pts[:,1])
    h = max_y-min_y
    min_x = np.min(contour_pts[:,0])
    max_x = np.max(contour_pts[:,0])
    w = max_x-min_x
    # compute average x-coordinates on two sides
    x_avg = np.mean(contour_pts[:,0])
    
    y_avg = np.mean(contour_pts[:,1])
    # separate the points on left boundary and on right boundary
    left_boundary_pts = contour_pts[(contour_pts[:,0]<x_avg) & (contour_pts[:,1]>min_y+0.03*h) & (contour_pts[:,1]<max_y-0.03*h)]
    right_boundary_pts = contour_pts[(contour_pts[:,0]>x_avg) & (contour_pts[:,1]>min_y+0.03*h) & (contour_pts[:,1]<max_y-0.03*h)] 
    
    bot_boundary_pts = contour_pts[(contour_pts[:,1]<y_avg) & (contour_pts[:,0]>min_x+0.03*w) & (contour_pts[:,0]<max_x-0.03*w)]
    top_boundary_pts = contour_pts[(contour_pts[:,1]>y_avg) & (contour_pts[:,0]>min_x+0.03*w) & (contour_pts[:,0]<max_x-0.03*w)]
    # Compute the mean x-coordinates of left and right boundary points, respectively
    x_mean_left_boundary = np.mean(left_boundary_pts[:,0])
    x_mean_right_boundary = np.mean(right_boundary_pts[:,0])
    
    y_mean_bot_boundary = np.mean(bot_boundary_pts[:,1])
    y_mean_top_boundary = np.mean(top_boundary_pts[:,1])
    # Get the width of the strip
    width_strip = x_mean_right_boundary - x_mean_left_boundary
    height_strip = y_mean_top_boundary - y_mean_bot_boundary
    return width_strip, height_strip

#%%
# ** Tuples are defined by commas ,, not necessarily by parentheses (). 
# A single element tuple must include a comma, even if enclosed in parentheses, like (1,).**

def img_folder_to_str_list(folder_path: Path) -> List:
    """Given a folder path. Will return the path(string) to all files in that path in order."""
    name_list = glob.glob(str(folder_path) + '/*.png') # adapt here for .TIF or .tiff
    name_list.sort()
    name_list_str = []
    for name in name_list:
        name_list_str.append(str(name))
    return name_list_str

def give_widths_list(image_folder:List):
    widthNheight_list = []
    for idx, img_path in enumerate(image_folder):
        widthNheight = find_width(img_path)
        widthNheight_list.append(widthNheight)
        
    return widthNheight_list
#%%
if __name__ == '__main__':
    raw_masks_path = Path(".\images\predict")
    figures_list = img_folder_to_str_list(raw_masks_path)
    widthNheight_list = give_widths_list(figures_list)
    # now write the width information into a CSV file
    csv_file_path = r".\results"
    csv_file_path = osp.join(csv_file_path, 'DNgel01_widths_UNet.csv')
    with open(csv_file_path,'w',newline='') as file:
        writer = csv.writer(file)
        # write each number on a new line
        for (width,height) in widthNheight_list:
            #writer.writerow([width,height])
            writer.writerow([width])
            
    print(f"file 'widths_UNet.csv' written successfully and save to {csv_file_path}")
    

