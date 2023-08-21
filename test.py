# import os
# import numpy as np
# import torch
# import time
#
# from nets.facenet import Facenet
#
# # def process_event(profiler_events):
#
#if __name__ == '__main__':
#     backbone="mobilenet"
#     num_classes=2
#     pretrained=False
#     watermark_size=128
#     model = Facenet(backbone=backbone, num_classes=num_classes, pretrained=pretrained, watermark_size=watermark_size,
#                     dropout_keep_prob=0.5)
#     device = torch.device('cuda')
#     model_path="/home/lsf/facenet-pytorch/model_data/facenet_mobilenet.pth"
#     if model_path != '':
#         # ------------------------------------------------------#
#         #   权值文件请看README，百度网盘下载
#         # ------------------------------------------------------#
#         print('Load weights {}.'.format(model_path))
#
#         # ------------------------------------------------------#
#         #   根据预训练权重的Key和模型的Key进行加载
#         # ------------------------------------------------------#
#         model_dict = model.state_dict()
#         pretrained_dict = torch.load(model_path, map_location=device)
#         load_key, no_load_key, temp_dict = [], [], {}
#         model_key=list(model_dict.keys())
#         i=0
#         for k, v in pretrained_dict.items():
#             if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
#                 temp_dict[k] = v
#                 load_key.append(k)
#             else:
#                 if "modulation" in model_key[i]:
#                     i+=2
#
#                 print("---")
#                 print(model_key[i])
#                 print(np.shape(model_dict[model_key[i]]))
#                 print(np.shape(v))
#                 print(("---"))
#                 if np.shape(model_dict[model_key[i]])==np.shape(v):
#                     temp_dict[model_key[i]] =v
#                     load_key.append(model_key[i])
#                 #no_load_key.append(k)
#             i+=1
#         print("*******")
#         for k in load_key:
#             print(k)
#         # model_dict.update(temp_dict)
#         # model.load_state_dict(model_dict)
import ast

import matplotlib.pyplot as plt
import numpy as np

# iteration1 = []
# Loss1 = []
# i=0
# with open('loss_train50w_0.01_v2.txt', 'r') as file:  # 打开文件
#     for line in file.readlines():  # 文件内容分析成一个行的列表
#         Loss1.append(line)
#         iteration1.append(i)
#         i +=1

if __name__ == '__main__':
    iteration1 = []
    acc1 = []
    i = 0
    with open('/root/autodl-tmp/facenet-pytorch/robustness/wm32_combine_wmacc.txt', 'r') as file:
        for line in file.readlines():
            line=float(line)
            acc1.append(line)
            iteration1.append(i)
            i += 0.05

    iteration2 = []
    acc2 = []
    i = 0
    with open('/root/autodl-tmp/facenet-pytorch/robustness/wm32_combine_LFWacc.txt', 'r') as file:
        for line in file.readlines():
            line = float(line)
            acc2.append(line)
            iteration2.append(i)
            i += 0.05

    iteration3 = []
    acc3 = []
    i = 0
    with open('/root/autodl-tmp/facenet-pytorch/robustness/LSB_combine_wmacc.txt', 'r') as file:
        for line in file.readlines():
            line = float(line)
            acc3.append(line)
            iteration3.append(i)
            i += 0.05

    # 画图
    # plt.title('Loss')  # 标题
    # plt.plot(iteration1, Loss1, color='cyan', label='loss_wm')
    # plt.legend()  # 显示上面的label
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')

    plt.title('combine')
    plt.plot(iteration1, acc1, 'b', label='WM32_acc')
    plt.plot(iteration2, acc2, 'red', label='LFW_acc')
    plt.plot(iteration3, acc3, 'green', label='LSBwm_acc')  # 'b'指：color='blue'
    plt.legend()  # 显示上面的label
    plt.xlabel('Combine')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.2)  # 仅设置y轴坐标范围
    plt.savefig('robustness/Combine.png')
    plt.show()
