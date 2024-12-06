# -*- coding: utf-8 -*-

"""
Author: Simeng Wu
Email: sw3493@drexel.edu
"""

import os
import time

from tqdm import tqdm
from utils.utils_metrics import compute_mIoU_gray, show_results
import glob
import numpy as np
import torch
import os
import cv2
from model.unet_model import UNet


def cal_miou(test_dir="../DRIVE-SEG-DATA/Test_Images",
             pred_dir="../DRIVE-SEG-DATA/results",
             gt_dir="../DRIVE-SEG-DATA/Test_Labels",
             model_path='best_model_drive.pth'):
    """Calculate the mean Intersection-over-Union (mIoU) and other metrics.

    This function loads a pre-trained UNet model and performs inference on a test dataset.
    It then computes segmentation metrics including mIoU, pixel accuracy, recall, and precision.
    The results are saved in the specified output directory.

    Args:
        test_dir (str): Path to the directory containing test images.
        pred_dir (str): Path to the directory where prediction results will be saved.
        gt_dir (str): Path to the directory containing ground truth (label) images.
        model_path (str): Path to the pre-trained model checkpoint (.pth file).

    Returns:
        None. The function writes prediction images to the output directory and saves computed metrics.
    """
    # Define class names for display and metrics computation
    name_classes = ["background", "vein"]
    num_classes = len(name_classes)

    # Create the prediction directory if it does not exist
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    print("---------------------------------------------------------------------------------------")
    print("Loading the trained model from: {}".format(model_path))

    # Determine the device: use GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the UNet model with 1 input channel and 1 output channel
    net = UNet(n_channels=1, n_classes=1)

    # Move the model to the chosen device
    net.to(device=device)

    # Load the trained model parameters
    net.load_state_dict(torch.load(model_path, map_location=device))

    # Set the model to evaluation mode
    net.eval()
    print("Model loaded successfully!")

    # List all test images and extract their base file names (IDs)
    img_names = os.listdir(test_dir)
    image_ids = [image_name.split(".")[0] for image_name in img_names]

    print("---------------------------------------------------------------------------------------")
    print("Starting batch inference on the test dataset...")
    time.sleep(1)

    # Perform inference on each test image
    for image_id in tqdm(image_ids):
        image_path = os.path.join(test_dir, image_id + ".png")

        # Read the test image
        img = cv2.imread(image_path)
        origin_shape = img.shape

        # Convert the image to grayscale as the network expects single-channel input
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Resize the image to the input size expected by the model (512x512)
        img = cv2.resize(img, (512, 512))

        # Reshape image to (batch=1, channel=1, height=512, width=512)
        img = img.reshape(1, 1, img.shape[0], img.shape[1])

        # Convert the numpy array to a PyTorch tensor
        img_tensor = torch.from_numpy(img).to(device=device, dtype=torch.float32)

        # Perform forward pass to get predictions
        pred = net(img_tensor)

        # Convert predictions to a numpy array and extract the predicted mask
        pred = np.array(pred.data.cpu()[0])[0]

        # Binarize predictions: threshold at 0.5
        # Pixels >= 0.5 are set to 255 (foreground), otherwise 0 (background)
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0

        # Resize the prediction back to the original image size
        pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)

        # Save the prediction mask
        cv2.imwrite(os.path.join(pred_dir, image_id + ".png"), pred)

    print("Batch inference completed.")
    print("Calculating evaluation metrics (mIoU, etc.)...")

    # Compute metrics including mIoU, IoU per class, pixel accuracy, recall, and precision
    hist, IoUs, PA_Recall, Precision = compute_mIoU_gray(
        gt_dir, pred_dir, image_ids, num_classes, name_classes
    )

    print("Metrics calculation completed. Results will be saved to the 'results' directory.")
    miou_out_path = "results/"

    # Display and save results
    show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)


if __name__ == '__main__':
    # The directories below can be customized as needed.
    # test_dir: directory containing test images
    # pred_dir: directory where prediction masks are saved
    # gt_dir: directory containing ground-truth masks
    # model_path: path to the trained model checkpoint file
    cal_miou(
        test_dir="../DRIVE-SEG-DATA/Test_Images",
        pred_dir="../DRIVE-SEG-DATA/results",
        gt_dir="../DRIVE-SEG-DATA/Test_Labels",
        model_path='best_model_drive.pth'
    )
