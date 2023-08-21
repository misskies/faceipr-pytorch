# encoding:utf-8

import cv2
import numpy as np
import os
import sys
from tqdm import tqdm

if __name__ == '__main__':
    # 要处理的图片路径
    img_path = '/home/lsf/celeba-datasets/img_celeba/'
    # 新图片存储路径
    new_img_path = '/home/lsf/celeba-datasets/CelebA_img/'
    # 人脸landmark标注文件地址
    landmark_anno_file_path = '/home/lsf/celeba/Anno/list_landmarks_celeba.txt'
    # 人脸bbox标注文件地址
    face_boundingbox_anno_file_path = '/home/lsf/celeba/Anno/list_bbox_celeba.txt'
    # 新的人脸landmark标注文件地址
    new_landmark_anno_file_path = '/home/lsf/celeba-datasets/Anno/new_list_landmarks_celeba.txt'

    # 新图片的高度及宽度
    new_h = 256
    new_w = 256

    if not os.path.exists(img_path):
        print("image path not exist.")
        exit(-1)

    if not os.path.exists(landmark_anno_file_path):
        print("landmark_anno_file not exist.")
        exit(-1)

    if not os.path.exists(face_boundingbox_anno_file_path):
        print("face_boundingbox_anno_file not exist.")
        exit(-1)

    if not os.path.exists(new_img_path):
        os.makedirs(new_img_path)

    # 加载文件
    landmark_anno_file = open(landmark_anno_file_path, 'r')
    face_boundingbox_anno_file = open(face_boundingbox_anno_file_path, 'r')
    new_landmark_anno_file = open(new_landmark_anno_file_path, 'w')
    landmark_anno = landmark_anno_file.readlines()
    face_bbox = face_boundingbox_anno_file.readlines()
    for i in tqdm(range(2, len(landmark_anno))):
        landmark_split = landmark_anno[i].split()
        face_bbox_split = face_bbox[i].split()
        filename = landmark_split[0]
        if filename != face_bbox_split[0]:
            print(filename, face_bbox_split[0])
            break
        landmark = []
        face = []
        for j in range(1, len(landmark_split)):
            landmark.append(int(landmark_split[j]))
        for j in range(1, len(face_bbox_split)):
            face.append(int(face_bbox_split[j]))
        landmark = np.array(landmark)
        landmarks = np.resize(landmark, (5, 2))
        face = np.array(face)

        try:
            path = os.path.join(img_path, filename)
            new_path = os.path.join(new_img_path, filename)
            if not os.path.exists(path):
                print(path, 'not exist')
                continue
            img = cv2.imread(path)

            # 裁剪图像
            newImg = img[face[1]:face[3] + face[1], face[0]:face[2] + face[0]]

            # 重新计算新的landmark坐标并存储
            new_landmark_str = ""
            new_landmark_str += filename + '\t'
            for landmark in landmarks:
                landmark[0] -= face[0]
                landmark[1] -= face[1]
                landmark[0] = round(landmark[0] * (new_w * 1.0 / newImg.shape[1]))
                landmark[1] = round(landmark[1] * (new_h * 1.0 / newImg.shape[0]))
                new_landmark_str += str(landmark[0]) + '\t' + str(landmark[1]) + '\t'
            new_landmark_str += '\n'
            new_landmark_anno_file.write(new_landmark_str)
            new_landmark_anno_file.flush()
            resizeImg = cv2.resize(newImg, (new_h, new_w))
            # 存储新图片
            cv2.imwrite(new_path, resizeImg)
        except:
            print("filename:%s process failed" % (filename))

    landmark_anno_file.close()
    face_boundingbox_anno_file.close()
    new_landmark_anno_file.close()
