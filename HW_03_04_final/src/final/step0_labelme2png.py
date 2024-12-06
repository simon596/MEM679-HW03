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
    print("准备开始json文件转化 注意json文件不用有中文")
    # delete all trees and contents in dir 'labels' from previous runs
    if osp.isdir(png_save_folder):
        shutil.rmtree(png_save_folder)
    os.makedirs(png_save_folder)
    print("数据保存目录创建成功！")
    # 遍历文件夹，把文件夹中得json文件夹先删除
    for json_file in os.listdir(json_folder):
        file_path = osp.join(json_folder, json_file)
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)
            print("文件夹清除成功！")
    print("开始批量生成json文件")
    for json_file in os.listdir(json_folder):
        json_path = osp.join(json_folder, json_file)
        # most important step. Execute a command in the system's shell
        os.system("labelme_json_to_dataset {}".format(json_path))
        # useful method to change extension name
        label_path = osp.join(json_folder, json_file.split(".")[0] + "_json/label.png")
        png_save_path = osp.join(png_save_folder, json_file.split(".")[0] + ".png")
        label_png = cv2.imread(label_path, 0)
        label_png[label_png > 0] = 255
        cv2.imwrite(png_save_path, label_png)
        # shutil.copy(label_path, png_save_path)
        # break
    print("标签文件已保存在目录：{}".format(png_save_folder))



if __name__ == '__main__':
    # !!!!你的json文件夹下只能有json文件不能有其他文件
    json2png(json_folder="labelme_test_data/jsons/", png_save_folder="labelme_test_data/labels/")
