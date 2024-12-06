# -*-coding:utf-8 -*-

"""
#-------------------------------
# Author: Simeng Wu
# Email: sw3493@drexel.edu
#-------------------------------

#-------------------------------
"""

import glob
import numpy as np
import torch
import os
import cv2
from model.unet_model import UNet

# The following code is executed as a script and is designed for making predictions on input images using a UNet model.
if __name__ == "__main__":
    # Specify the directory where prediction results will be saved
    save_dir = "images/predict"
    # Choose the device to use for prediction; 'cuda' if available, else 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Currently using device: {}'.format(device))
    
    # Initialize the UNet model with single input channel (grayscale) and single output class (binary segmentation)
    net = UNet(n_channels=1, n_classes=1)
    # Move the model to the selected device
    net.to(device=device)
    
    # Load the saved model parameters
    net.load_state_dict(torch.load('best_model.pth', map_location=device))
    
    # Set the model to evaluation mode for prediction
    net.eval()
    
    # Retrieve a list of all PNG images in the specified directory
    tests_path = glob.glob('../raw_png/*.png')
    
    # Process each image file
    for i, test_path in enumerate(tests_path):
        # Construct the path to save the prediction result
        save_res_path = os.path.join(save_dir, os.path.basename(test_path))
        
        # Read the input image
        img = cv2.imread(test_path)
        origin_shape = img.shape
        
        # Convert the input image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Resize the image to match the model input size (512 x 512)
        img = cv2.resize(img, (512, 512))
        
        # Reshape the image to include batch and channel dimensions: (batch=1, channel=1, width=512, height=512)
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        
        # Convert the NumPy array to a PyTorch tensor
        img_tensor = torch.from_numpy(img)
        
        # Move the tensor to the chosen device and ensure it's a float
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        
        # Perform prediction using the model
        pred = net(img_tensor)
        
        # Convert the prediction output to a NumPy array and extract the first batch and channel
        pred = np.array(pred.data.cpu()[0])[0]
        
        # Binarize the prediction results: values >= 0.5 become 255 (white), otherwise 0 (black)
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        
        # Resize the prediction back to the original image size
        pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Save the prediction mask as an image
        cv2.imwrite(save_res_path, pred)
        print("{}: The prediction result for {} has been saved to {}".format(i+1, test_path, save_res_path))
