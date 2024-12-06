import cv2
import numpy as np
import glob
import os.path as osp

def overlay_bdry(original_img_path, mask_path,  output_folder):
    # Load the original image
    original_img = cv2.imread(original_img_path)

    # Load the prediction mask (ensure it's in grayscale)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Find contours from the binary image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Identify the largest contour based on the area
    largest_contour = max(contours, key=cv2.contourArea)

    # Draw the largest contour on the original image
    # cv2.drawContours(destination image, contours, contourIdx, color, thickness)
    # contourIdx = -1 to draw all contours, here we pass the largest_contour inside a list
    # color in BGR (e.g., (0, 255, 0) for green), thickness of the lines
    cv2.drawContours(original_img, [largest_contour], -1, (0, 255, 0), 2)

    # Display the image with the largest mask boundary
    #cv2.imshow('Image with Largest Mask Boundary', original_img)
    #cv2.waitKey(0)  # Wait for a key press to exit
    #cv2.destroyAllWindows()

    # Optionally, save the result to a file
    cv2.imwrite(osp.join(output_folder, osp.basename(original_img_path) + ".png"), original_img)
    
if __name__ == '__main__':
    img_folder = r"C:\Users\sw3493\Python-Proj\Greg-acetic-06112024\raw_png"
    mask_folder = r"C:\Users\sw3493\Python-Proj\Greg-acetic-06112024\indentation_test\images\predict"
    output_folder = r"C:\Users\sw3493\Python-Proj\Greg-acetic-06112024\indentation_test\images\overlay_bdry"
    all_img_path = glob.glob(img_folder + '/*.png')
    
    for i, img_path in enumerate(all_img_path):
        mask_path = osp.join(mask_folder, osp.basename(img_path))
        overlay_bdry(img_path,mask_path,output_folder)
