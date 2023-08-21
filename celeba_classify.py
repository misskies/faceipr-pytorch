# -*- coding: utf-8 -*-
# !/usr/bin/env python3

'''
Divide face accordance CelebA Id type.
'''


import shutil
import os

output_path_train = r"/home/lsf/celeba_classify"
image_path = r"/home/lsf/celeba-datasets/CelebA_img"  #原始图片文件夹的路径
CelebA_Id_file = r"/home/lsf/celeba/Anno/identity_CelebA.txt"  #identity_CelebA.txt文件的路径


def main():
    count_N = 0

    with open(CelebA_Id_file, "r") as Id_file:

        Id_info = Id_file.readlines()
        for line in Id_info:
            count_N += 1   #计数
            info = line.split()
            filename = info[0]
            file_Id = info[1]
            Id_dir_train = os.path.join(output_path_train,file_Id)
            filepath_old = os.path.join(image_path,filename) #原始照片所在的位置
            if not os.path.isdir(Id_dir_train):
                os.makedirs(Id_dir_train)
            else:
                pass
            train = os.path.join(Id_dir_train,filename)
            if filename == "101283.jpg" :
                print("pass")
            else:
                shutil.copyfile(filepath_old,train)        #这句代码是复制的意思
    Id_file.close()
    print(" have %d images!" % count_N)

if __name__ == "__main__":
    main()
