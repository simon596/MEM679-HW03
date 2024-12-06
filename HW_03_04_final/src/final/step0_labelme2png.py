# -*-coding:utf-8 -*-

"""
#-------------------------------
# Author: Simeng Wu
# Email: sw3493@drexel.edu
#-------------------------------
# Creation Date: Feb 28th, 2024, EST.
#-------------------------------
"""

from __future__ import print_function
import argparse
import glob
import math
import json
import os
import os.path as osp
import shutil
import numpy as np
import PIL.Image
import PIL.ImageDraw
import cv2


def json2png(json_folder, png_save_folder):
    """Converts a folder of JSON annotation files to PNG label images.

    This function takes a directory containing JSON annotation files 
    created by labelme, converts them into PNG label masks, and saves 
    them to the specified output directory. The output PNG files will 
    have the same filename as their corresponding JSON files, but with 
    a ".png" extension.

    Process:
        1. Removes any existing PNG save directory and its contents, 
           then recreates it.
        2. Removes any subdirectories inside the JSON folder.
        3. For each JSON file, executes the `labelme_json_to_dataset` 
           command line tool to create a temporary dataset folder.
        4. Extracts the "label.png" from the dataset folder, binarizes 
           it (labels > 0 are set to 255), and saves it to the PNG 
           output directory.

    Args:
        json_folder (str): The path to the folder containing the JSON 
            annotation files. The folder should only contain JSON files.
        png_save_folder (str): The path to the folder where the 
            converted PNG label images will be saved.

    Raises:
        OSError: If the output directory or intermediate directories 
            cannot be created or removed.
        RuntimeError: If the `labelme_json_to_dataset` command fails 
            (not explicitly checked, but will fail if command is missing).

    Side Effects:
        - Removes and recreates the output directory.
        - Removes any subdirectories found in `json_folder`.
        - Executes a system command (`labelme_json_to_dataset`).
        - Writes PNG label files into `png_save_folder`.

    Example:
        json2png(json_folder="labelme_test_data/jsons/", 
                 png_save_folder="labelme_test_data/labels/")
    """
    print("Starting to convert JSON files to PNG. Note: JSON files should not contain Chinese characters.")

    # Remove existing output directory if present and create a new one
    if osp.isdir(png_save_folder):
        shutil.rmtree(png_save_folder)
    os.makedirs(png_save_folder)
    print("Output directory successfully created!")

    # Remove any subdirectories inside the JSON folder
    for json_file in os.listdir(json_folder):
        file_path = osp.join(json_folder, json_file)
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)
            print("Subdirectory removed successfully!")
    print("Beginning batch JSON file processing.")

    # Process each JSON file
    for json_file in os.listdir(json_folder):
        json_path = osp.join(json_folder, json_file)
        
        # Execute the labelme command to generate dataset folder
        os.system("labelme_json_to_dataset {}".format(json_path))

        # Construct paths for the generated label image and the new PNG output
        label_path = osp.join(json_folder, json_file.split(".")[0] + "_json/label.png")
        png_save_path = osp.join(png_save_folder, json_file.split(".")[0] + ".png")

        # Read the label image in grayscale
        label_png = cv2.imread(label_path, 0)
        
        # Binarize the label: set all non-zero values to 255
        label_png[label_png > 0] = 255
        
        # Save the binarized label image
        cv2.imwrite(png_save_path, label_png)

    print("Label files have been saved to: {}".format(png_save_folder))


if __name__ == '__main__':
    # Note: The JSON folder should only contain JSON files and no other file types.
    json2png(json_folder="labelme_test_data/jsons/", png_save_folder="labelme_test_data/labels/")
