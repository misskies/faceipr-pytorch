import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
from nets.facenet import Facenet, Facenet_loss
from utils.dataloader import LFWDataset
from utils.utils_metrics import test, LSB_test, post_test, loss_baseline_test

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--input_shape', type=list, default=[160, 160, 3], help='input shape')
    parse.add_argument('--batch_size', type=int, default=128, help='batch size')
    parse.add_argument('--backbone', type=str, default='mobilenet', help='backbone')
    parse.add_argument('--model_path', type=str,
                       default='',
                       help='load model from checkpoint')

    parse.add_argument('--watermark_size', type=int, default=32, help='watermark size')

    parse.add_argument('--lfw_dir_path', type=str, default='lfw', help='lfw dir path')
    parse.add_argument('--lfw_pairs_path', type=str, default='model_data/lfw_pair.txt', help='lfw pairs path')

    parse.add_argument('--original', type=bool, default=False, help='Whether modulate the mode')

    parse.add_argument('--robustness', type=str, default='none', help='',
                       choices=['none', 'noise', 'flip', 'round', 'random_del', 'combine'])

    parse.add_argument('--png_save_path', type=str, default='model_data/roc_test.png', help='Roc save path')

    # parse.add_argument('--LSB', type=bool, default=False, help='Eval LSB Baseline')
    parse.add_argument("--post", type=str, default="None", choices=["None", "LSB", "FFT", "Noise"],
                       help='Eval LSB Baseline')
    parse.add_argument("--test_rank", type=int, default="21", help='eval rank')
    parse.add_argument("--noise_power", type=float, default="0", help='noise power')
    parse.add_argument('--loss_baseline', type=bool, default=False, help='Whether eval loss_baseline')
    parse.add_argument('--loss_baseline_wm_path', type=str, default="/home/lsf/public/collaboration/facenet-pytorch/facenet-pytorch/trained_weight/faceweb_lossEmbed_mobilenet/loss_baseline_watermark_in.pt", help='The fixed watermark for loss_baseline')
    

    args = parse.parse_args()
    # --------------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # --------------------------------------#
    cuda = True
    # --------------------------------------#
    #   主干特征提取网络的选择
    #   mobilenet
    #   inception_resnetv1
    # --------------------------------------#
    backbone = args.backbone
    # --------------------------------------------------------#
    #   输入图像大小，常用设置如[112, 112, 3]
    # --------------------------------------------------------#ss
    input_shape = args.input_shape
    # --------------------------------------#
    #   训练好的权值文件
    # --------------------------------------#
    # LSB = args.LSB
    post = args.post
    # --------------------------------------#
    #   评估LSB Baseline
    # --------------------------------------#
    model_path = args.model_path
    # --------------------------------------#
    #   LFW评估数据集的文件路径
    #   以及对应的txt文件

    # --------------------------------------#
    lfw_dir_path = args.lfw_dir_path
    lfw_pairs_path = args.lfw_pairs_path
    # --------------------------------------#
    #   评估的批次大小和记录间隔
    # --------------------------------------#
    batch_size = args.batch_size
    log_interval = 1
    watermark_size = args.watermark_size
    robustness = args.robustness
    original = args.original
    # --------------------------------------#
    #   ROC图的保存路径
    # --------------------------------------#
    png_save_path = args.png_save_path
    loss_baseline = args.loss_baseline

    test_loader = torch.utils.data.DataLoader(
        LFWDataset(dir=lfw_dir_path, pairs_path=lfw_pairs_path, image_size=input_shape), batch_size=batch_size,
        shuffle=False)

    test_rank = args.test_rank
    noise_power = args.noise_power
    for i in range(test_rank):
        if i == 0:
            robustness = "none"
        else:
            robustness = args.robustness
            if robustness == "round":
                noise_power -= 1
            else:
                noise_power += 0.05
        if loss_baseline:
            model = Facenet_loss(backbone=backbone,mode="predict",
                                 dropout_keep_prob=0.5,robustness=robustness,noise_power=noise_power)
        else:
            model = Facenet(backbone=backbone, mode="predict", watermark_size=watermark_size, robustness=robustness,
                        noise_power=noise_power)

        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                if original and "Encoder" in k and "Decoder" in k:
                    no_load_key.append(k)
                else:
                    temp_dict[k] = v
                    load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        model = model.eval()
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

        if cuda:
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model = model.cuda()
        if post != "None":
            # watermark_size=1024
            # LSB_test(test_loader, model, png_save_path, log_interval, batch_size, cuda, watermark_size)
            post_test(test_loader, model, png_save_path, log_interval, batch_size, cuda, args.watermark_size,
                      post_method=post, robustness=robustness, noise_power=noise_power,test_robustness=args.robustness)
        else:
            if loss_baseline:
                loss_baseline_test(test_loader, model, png_save_path, log_interval, batch_size, cuda, watermark_size,robustness,noise_power,test_robustness=args.robustness, wm_path=args.loss_baseline_wm_path)
            else:
                test(test_loader, model, png_save_path, log_interval, batch_size, cuda, watermark_size, original,robustness,noise_power,test_robustness=args.robustness)