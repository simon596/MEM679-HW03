import csv
import numpy as np
import matplotlib.pyplot as plt
import cv2

def volume(width, height):
    """Units: pixel number"""
    volume = width**2*height
    return volume

# fill width list
widths_list = []
with open(r".\results\DNgel01_widths_UNet.csv",'r') as widths_csv:
    data2 = csv.reader(widths_csv)
    for row_num, row in enumerate(data2):
        if row_num>=0 and row_num<=689:
            try: 
                widths_list.append(float(row[0]))
            except Exception as e:
                print('something wrong happened')
                continue
        else:
            continue
        
# get fps
video_path = r"E:\Exponent\Maria_0722\SAMPLE1_01.avi"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
else:
    # Get the FPS of the video, which is used in interpolation
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frames Per Second (FPS): {fps}")
cap.release()

# height given by time*constant rate
const_rate = 0.15 # mm/s
init_height = 12.5 # mm
heights_list = [init_height - const_rate*(frame/fps) for frame in range(len(widths_list))]
# compute J_F by iteration
J_F_list = []
for iter, (w,h) in enumerate(zip(widths_list,heights_list)):
    if iter == 0: 
        init_vol = volume(w,h)
    J_F_list.append(volume(w,h)/init_vol)



strain_list = [0.15*(frame/fps) / 12.5 for frame in range(len(widths_list))]

# plot
plt.figure('J_F', figsize=(10,10))
plt.plot(strain_list, J_F_list, label='det(F)')
plt.xlabel('Strain (%)')
plt.ylabel('det(F)')
plt.title('Volume Ratio Change with Strain (by U-Net)')
plt.legend()
plt.show()
      
# write the noisy data into a CSV file  
with open(r".\results\profile.csv", 'w', newline='') as profile:
    writer = csv.writer(profile)
    for strain, J_F in zip(strain_list, J_F_list):
        writer.writerow([strain, J_F])
        

