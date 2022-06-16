import numpy as np
import scipy.io
import os
import cv2
import shutil
import matplotlib.pyplot as plt

input_folder_name = 'dataset/annotations'
output_folder_name = 'dataset/annotationsImages'
FileNameList = os.listdir(input_folder_name)

for i in range(len(FileNameList)):
    #  判断当前文件是否为json文件
    if (os.path.splitext(FileNameList[i])[1] == ".mat"):
        #  复制jpg文件 why??????
        # JPG_file = JPG_folder + "\\" + FileNameList[i]
        # new_JPG_file = Paste_JPG_folder + "\\" + str(NewFileName) + ".jpg"
        # shutil.copyfile(JPG_file, new_JPG_file)
        currentFile = scipy.io.loadmat(input_folder_name + '/' +FileNameList[i])
        file_name = FileNameList[i].split(".", 1)[0]
        file_name = file_name+".jpg"
        labelImage = currentFile['groundtruth']
        # plt.imsave(output_folder_name+"/"+file_name,labelImage)
        labelImage.dtype = 'uint8'
        cv2.imwrite(output_folder_name+"/"+file_name, labelImage)


