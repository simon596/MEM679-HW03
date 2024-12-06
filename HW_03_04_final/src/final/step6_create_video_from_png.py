# -*-coding:utf-8 -*-

"""
#-------------------------------
# Author: Simeng Wu
# Email: sw3493@drexel.edu
#-------------------------------

#-------------------------------
"""

import cv2
import os
import glob

# Specify the directory containing the images
image_folder = r"C:\Users\sw3493\Python-Proj\Greg-acetic-06112024\indentation_test\images\overlay_bdry"
video_name = r"C:\Users\sw3493\Python-Proj\Greg-acetic-06112024\DN_Gel01.mp4"

images = sorted(glob.glob(f'{image_folder}/*.png'))
frame = cv2.imread(images[0])
height, width, layers = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
video = cv2.VideoWriter(video_name, fourcc, 20.0, (width,height))

for image in images:
    video.write(cv2.imread(image))

cv2.destroyAllWindows()
video.release()
