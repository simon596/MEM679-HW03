# -*-coding:utf-8 -*-

"""
#-------------------------------
# Author: Simeng Wu
# Email: sw3493@drexel.edu
#-------------------------------

#-------------------------------
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import cv2

def volume(width, height):
    """
    Calculate the volume of a rectangular prism-like section based on width and height.
    Note: Units here are in terms of pixel counts; no unit conversion is performed.

    Args:
        width (float): The width of the section (in pixels).
        height (float): The height of the section (in pixels).

    Returns:
        float: The computed volume (width^2 * height, in pixel units).
    """
    volume_value = width**2 * height
    return volume_value

# Load widths from a CSV file
widths_list = []
with open(r".\results\DNgel01_widths_UNet.csv", 'r') as widths_csv:
    data2 = csv.reader(widths_csv)
    # Reading each row and extracting the width value
    for row_num, row in enumerate(data2):
        # Using a range condition (0 <= row_num <= 689) to limit the number of data points
        if 0 <= row_num <= 689:
            try:
                # Convert the read width value to float
                widths_list.append(float(row[0]))
            except Exception as e:
                print('Error: Unable to parse width value.')
                continue
        else:
            continue

# Load the video to determine frames per second (FPS)
video_path = r"E:\Exponent\Maria_0722\SAMPLE1_01.avi"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
else:
    # Get the FPS of the video, which will be used for time-based calculations
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frames Per Second (FPS): {fps}")
cap.release()

# Compute heights for each frame based on a constant rate of height reduction over time
# Assuming:
#   const_rate: the rate at which height decreases (mm/s)
#   init_height: the initial height in mm
#   frame/fps: converts frame number to seconds
const_rate = 0.15  # mm/s
init_height = 12.5 # mm
heights_list = [init_height - const_rate * (frame/fps) for frame in range(len(widths_list))]

# Compute the volume ratio J_F for each frame relative to the initial volume
J_F_list = []
for i, (w, h) in enumerate(zip(widths_list, heights_list)):
    if i == 0:
        # The initial volume is that of the first frame
        init_vol = volume(w, h)
    # J_F is defined as the current volume divided by the initial volume
    J_F_list.append(volume(w, h) / init_vol)

# Compute strain as a function of time. Assuming:
#   strain = 0.15*(time)/12.5, with time in seconds and initial height 12.5 mm
strain_list = [0.15 * (frame/fps) / 12.5 for frame in range(len(widths_list))]

# Plot the volume ratio (J_F) against the strain
plt.figure('J_F', figsize=(10,10))
plt.plot(strain_list, J_F_list, label='det(F)')
plt.xlabel('Strain (%)')
plt.ylabel('det(F)')
plt.title('Volume Ratio Change with Strain (by U-Net)')
plt.legend()
plt.show()

# Save the strain and J_F data to a CSV file for further analysis
with open(r".\results\profile.csv", 'w', newline='') as profile:
    writer = csv.writer(profile)
    # Write each pair of (strain, J_F) into the CSV file
    for strain, J_F in zip(strain_list, J_F_list):
        writer.writerow([strain, J_F])
