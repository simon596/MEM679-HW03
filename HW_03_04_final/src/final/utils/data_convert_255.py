# conversion of dataset
# change 0-255 to 0-1
import os
import os.path as osp
import cv2


# change colors
def data_convert(src_folder, target_folder):
    if osp.isdir(target_folder) == False:
        os.mkdir(target_folder)
    image_names = os.listdir(src_folder)
    for image_name in image_names:
        image_path = osp.join(src_folder, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img[img > 0] = 255
        # img[img==255] = 2
        save_path = osp.join(target_folder, image_name)
        cv2.imwrite(save_path, img)


if __name__ == '__main__':
    data_convert(src_folder=r"G:\AAA-projects\ING\unet-drive\DRIVE-SEG-DATA\Training_Labels_src",
                 target_folder=r"G:\AAA-projects\ING\unet-drive\DRIVE-SEG-DATA\Training_Labels")
