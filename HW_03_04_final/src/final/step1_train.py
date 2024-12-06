# -*-coding:utf-8 -*-

"""
#-------------------------------
# Author: Simeng Wu
# Email: sw3493@drexel.edu
#-------------------------------

#-------------------------------
"""

from model.unet_model import UNet
from utils.dataset import Dateset_Loader
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
import time
import matplotlib.pyplot as plt


def train_net(net, device, data_path, epochs=40, batch_size=1, lr=0.00001):
    """
    Train a given semantic segmentation model on a specified dataset.

    Args:
        net (nn.Module): The semantic segmentation network to be trained.
        device (torch.device): The device to be used for training (e.g., CPU or GPU).
        data_path (str): Path to the dataset.
        epochs (int): Number of training epochs.
        batch_size (int): Size of each training batch.
        lr (float): Learning rate.

    Returns:
        None
    """

    # Load the dataset
    dataset = Dateset_Loader(data_path)
    per_epoch_num = len(dataset) / batch_size
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    # Define the optimizer. Here we use the Adam optimizer. Previously, RMSProp was mentioned but commented out.
    # Adam is chosen for its adaptive learning rate and often good performance.
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-08, amsgrad=False)

    # Define the loss function. Here we use binary cross-entropy with logits loss.
    criterion = nn.BCEWithLogitsLoss()

    # Initialize the best loss as infinity to track improvement over epochs.
    best_loss = float('inf')

    # List to record loss values over epochs
    loss_record = []

    # The tqdm progress bar shows progress for total iterations over all epochs.
    with tqdm(total=epochs * per_epoch_num) as pbar:
        for epoch in range(epochs):
            # Set the network in training mode
            net.train()
            # Iterate over batches
            for image, label in train_loader:
                optimizer.zero_grad()

                # Transfer data to the specified device
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)

                # Forward pass
                pred = net(image)

                # Calculate loss
                loss = criterion(pred, label)

                # Update the progress bar description with the current epoch and loss
                pbar.set_description("Processing Epoch: {} Loss: {}".format(epoch + 1, loss))

                # Save the model if the current loss is better than the previously recorded best
                if loss < best_loss:
                    best_loss = loss
                    torch.save(net.state_dict(), 'best_model.pth')

                # Backpropagate and update weights
                loss.backward()
                optimizer.step()

                # Update the progress bar
                pbar.update(1)

            # Record the loss after each epoch
            loss_record.append(loss.item())

    # Plot the training loss curve
    plt.figure()
    plt.plot([i + 1 for i in range(0, len(loss_record))], loss_record)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('results/training_loss.png')


if __name__ == "__main__":
    # Select device. Use GPU if available, otherwise CPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set input channels and output classes. Here, n_classes=1 means binary segmentation.
    net = UNet(n_channels=1, n_classes=1)

    # Move the network to the device
    net.to(device=device)

    # Specify the dataset path and start training
    data_path = "../NEW-SEG-DATA"  # You can also use a relative path
    print("If the progress bar seems stuck, it might be computing. Please wait patiently.")
    time.sleep(1)

    # Start training. If GPU memory is insufficient, it will use CPU.
    train_net(net, device, data_path, epochs=40, batch_size=1)
