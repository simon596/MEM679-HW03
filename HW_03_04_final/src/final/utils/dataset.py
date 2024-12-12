import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
import numpy as np

class Dateset_Loader(Dataset):
    def __init__(self, data_path):
        # init function, read all pic under data_path
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'Training_Images/*.png')) # todo can change extension name

    def augment(self, image, flipCode):
        # use cv2.flip to enhance data quality, filpCode = 1: horizontal flip; 0: vertical flip; -1: horizon+vertical flip
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        # read pictures based on index
        image_path = self.imgs_path[index]
        # generate label_path based on image_path
        label_path = image_path.replace('Training_Images', 'Training_Labels')
        # label_path = label_path.replace('.png', '_manual1.png') 
        label_path = label_path.replace('.png', '.png') 

        #
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        image = cv2.resize(image, (512, 512))
        label = cv2.resize(label, (512, 512), interpolation=cv2.INTER_NEAREST)
        # 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        # 
        if label.max() > 1:
            label = label / 255
        # 
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        return image, label

    def __len__(self):
        # 
        return len(self.imgs_path)


class ISBI_Loader_RGB(Dataset):
    def __init__(self, data_path):
        # 
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'Training_Images/*.png'))
        # self.num_classes = num_classes

    def augment(self, image, flipCode):
        # 
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        #
        image_path = self.imgs_path[index]
        # 
        label_path = image_path.replace('Training_Images', 'Training_Labels')
        label_path = label_path.replace('.png', '_manual1.png')  #
        image = cv2.imread(image_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (512, 512))
        label = cv2.resize(label, (512, 512), interpolation=cv2.INTER_NEAREST)
        #
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape(3, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])

        return image, label

    def __len__(self):
        #
        return len(self.imgs_path)


if __name__ == "__main__":
    isbi_dataset = ISBI_Loader_RGB(r"F:\BBBBBB\Unet-Eye-nope\DRIVE-SEG-DATA")
    print("number of data: ", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=2,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)
