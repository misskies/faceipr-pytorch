import os
import string
from random import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torch.version import cuda
from torchvision.transforms import ToTensor

from nets.facenet import Facenet, Facenet_loss, Facenet_128
from nets.facenet_training import (get_lr_scheduler, set_optimizer_lr,
                                   triplet_loss, weights_init, watermark_loss)
from utils.callback import LossHistory
from utils.dataloader import FacenetDataset, LFWDataset, dataset_collate, FaceWebDataset
from utils.utils import get_num_classes, show_config
from utils.utils_fit import fit_one_epoch
import argparse
from utils.utils_original_fit import original_fit_one_epoch


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    print(torch.cuda.device_count())
    print(torch.__version__)
    print(torch.cuda.is_available())
    # torch.autograd.set_detect_anomaly(True)

    parse = argparse.ArgumentParser()
    parse.add_argument('--seed', type=int, default=20, help='random seed')
    parse.add_argument('--input_shape', type=list, default=[160, 160, 3], help='input shape')
    parse.add_argument('--batch_size', type=int, default=72, help='batch size')
    parse.add_argument('--num_workers', type=int, default=4, help='num workers')
    parse.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parse.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parse.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parse.add_argument('--optimizer_type', type=str, default='adam', help='optimizer type')
    # parse.add_argument('--scheduler_type', type=str, default='cosin', help='scheduler type')

    parse.add_argument('--backbone', type=str, default='mobilenet', help='backbone')
    parse.add_argument('--pretrained', type=bool, default=False, help='pretrained')
    parse.add_argument('--model_path', type=str,
                       default='/home/lsf/facenet-pytorch/trained_weight/faceweb_unmd_mobilenet/ep042-loss0.410-val_loss2.268.pth',
                       help='load model from checkpoint')
    parse.add_argument('--decoder_arch', type=str, default='fc', choices=['fc', 'conv', 'attn', 'conv_attn'],
                       help='The architecuture of decoder')

    parse.add_argument('--watermark_size', type=int, default=32, help='watermark size')
    parse.add_argument('--annotation_path', type=str, default='embedding_extract_cls_train.txt', help='annotation path')
    parse.add_argument('--save_dir', type=str, default='logs', help='save dir')
    parse.add_argument('--save_period', type=int, default=1, help='save period')

    parse.add_argument('--lfw_dir_path', type=str, default='lfw', help='lfw dir path')
    parse.add_argument('--lfw_pairs_path', type=str, default='model_data/lfw_pair.txt', help='lfw pairs path')

    parse.add_argument('--original', type=bool, default=False, help='Whether modulate the mode')

    parse.add_argument('--loss_baseline', type=bool, default=False, help='Whether train loss_baseline')
    parse.add_argument('--loss_baseline_lambda', type=float, default=1.0,
                       help='the lambda watermark value in loss baseline')

    parse.add_argument('--robustness', type=str, default='none', help='',
                       choices=['none', 'noise', 'flip', 'round', 'random_del', 'combine'])
    parse.add_argument('--noise_power', type=float, default=0.1, help='Noise injection power')

    parse.add_argument('--embed_128', default=False, action="store_true", help='Whether use 128 demension embedding')

    parse.add_argument('--local_rank', type=int, default=0, help='local rank')

    args = parse.parse_args()

    # -------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # -------------------------------#
    Cuda = True
    # ---------------------------------------------------------------------#
    #   distributed     用于指定是否使用单机多卡分布式运行
    #                   终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡。
    #                   Windows系统下默认使用DP模式调用所有显卡，不支持DDP。
    #   DP模式：
    #       设置            distributed = False
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP模式：
    #       设置            distributed = True
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    # ---------------------------------------------------------------------#
    distributed = False
    # ---------------------------------------------------------------------#
    #   sync_bn     是否使用sync_bn，DDP模式多卡可用
    # ---------------------------------------------------------------------#
    sync_bn = False
    # ---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存、需要pytorch1.7.1以上
    # ---------------------------------------------------------------------#
    fp16 = False
    # --------------------------------------------------------#
    #   是否只训练原始任务(是否加入Encoder-Decoder,以及调制卷积层)
    # --------------------------------------------------------#
    original = args.original
    # --------------------------------------------------------#
    #   是否训练loss_function_baseline
    # --------------------------------------------------------#
    loss_baseline = args.loss_baseline
    # --------------------------------------------------------#
    #   指向根目录下的cls_train.txt，读取人脸路径与标签
    # --------------------------------------------------------#
    annotation_path = args.annotation_path
    # --------------------------------------------------------#
    #   输入图像大小，常用设置如[112, 112, 3]
    # --------------------------------------------------------#
    input_shape = args.input_shape
    # --------------------------------------------------------#
    #   主干特征提取网络的选择
    #   mobilenet
    #   inception_resnetv1
    # --------------------------------------------------------#
    backbone = args.backbone
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的训练的参数，来保证模型epoch的连续性。
    #
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的，pretrain不影响此处的权值加载。
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path = ''，pretrain = True，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，pretrain = Fasle，此时从0开始训练。
    # ----------------------------------------------------------------------------------------------------------------------------#
    model_path = args.model_path
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
    #   如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
    #   如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
    #   如果不设置model_path，pretrained = False，此时从0开始训练。
    # ----------------------------------------------------------------------------------------------------------------------------#
    pretrained = args.pretrained

    # ----------------------------------------------------------------------------------------------------------------------------#
    #   显存不足与数据集大小无关，提示显存不足请调小batch_size。
    #   受到BatchNorm层影响，不能为1。
    #
    #   在此提供若干参数设置建议，各位训练者根据自己的需求进行灵活调整：
    #   （一）从预训练权重开始训练：
    #       Adam：
    #           Init_Epoch = 0，Epoch = 100，optimizer_type = 'adam'，Init_lr = 1e-3，weight_decay = 0。
    #       SGD：
    #           Init_Epoch = 0，Epoch = 100，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 5e-4。
    #       其中：UnFreeze_Epoch可以在100-300之间调整。
    #   （二）batch_size的设置：
    #       在显卡能够接受的范围内，以大为好。显存不足与数据集大小无关，提示显存不足（OOM或者CUDA out of memory）请调小batch_size。
    #       受到BatchNorm层影响，batch_size最小为2，不能为1。
    #       正常情况下Freeze_batch_size建议为Unfreeze_batch_size的1-2倍。不建议设置的差距过大，因为关系到学习率的自动调整。
    # ----------------------------------------------------------------------------------------------------------------------------#
    # ------------------------------------------------------#
    #   训练参数
    #   Init_Epoch      模型当前开始的训练世代
    #   batch_size      每次输入的图片数量
    #                   受到数据加载方式与triplet loss的影响
    #                   batch_size需要为3的倍数
    #   Epoch           模型总共训练的epochss
    # ------------------------------------------------------#
    batch_size = args.batch_size
    Init_Epoch = 0
    Epoch = 1

    # ------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    # ------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    # ------------------------------------------------------------------#
    Init_lr = args.lr
    Min_lr = Init_lr * 0.01
    # ------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    # ------------------------------------------------------------------#
    optimizer_type = args.optimizer_type
    momentum = args.momentum
    weight_decay = args.weight_decay
    # ------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
    # ------------------------------------------------------------------#
    lr_decay_type = "cos"
    # ------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值，默认每个世代都保存
    # ------------------------------------------------------------------#
    save_period = args.save_period
    # ------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    # ------------------------------------------------------------------#
    save_dir = args.save_dir
    # ------------------------------------------------------------------#
    #   用于设置是否使用多线程读取数据
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   内存较小的电脑可以设置为2或者0
    # ------------------------------------------------------------------#
    num_workers = args.num_workers
    # ------------------------------------------------------------------#
    #   是否开启LFW评估
    # ------------------------------------------------------------------#
    lfw_eval_flag = True
    # ------------------------------------------------------------------#
    #   LFW评估数据集的文件路径和对应的txt文件
    # ------------------------------------------------------------------#
    lfw_dir_path = args.lfw_dir_path
    lfw_pairs_path = args.lfw_pairs_path

    # ------------------------------------------------------#
    #   设置用到的显卡
    # ------------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank = 0
    rank = 0

    # ---------------------------------#
    #   载入模型并加载预训练权重
    # ---------------------------------#
    # init_type="kaiming"
    watermark_size = args.watermark_size

    model = Facenet(backbone=backbone,  pretrained=pretrained,mode = "predict",
                    watermark_size=watermark_size, dropout_keep_prob=0.5, robustness=args.robustness,
                    noise_power=args.noise_power, decoder_arch=args.decoder_arch)
    # weights_init(model,init_type)
    if model_path != '':
        # ------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        # ------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        # ------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        # ------------------------------------------------------#
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                if ("Encoder" in k or "Decoder" in k) and original:
                    no_load_key.append(k)
                else:
                    temp_dict[k] = v
                    load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # ------------------------------------------------------#
        #   显示没有匹配上的Key
        # ------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
    scaler = None
    print(device)
    model = model.to(device)
    with open(annotation_path,"r") as f:
        lines = f.readlines()


    if True:
        if batch_size % 3 != 0:
            raise ValueError("Batch_size must be the multiple of 3.")
        # ---------------------------------------#
        #   构建数据集加载器。
        # ---------------------------------------#
        dataset = FaceWebDataset(lines,transform = ToTensor())
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=False)

        num = 0
        for iteration ,batch in enumerate(dataloader):
            tensor_list = []
            images = batch
            if cuda :
                images = images.cuda(local_rank)
            embedding  = model(images,None,"original_predict")
            tmp_list = torch.split(embedding,1)
            num +=images.shape[0]
            for  i in tmp_list:
                tensor_list.append(i.view(128))
            torch.save(tensor_list, f'/home/lsf/facenet-pytorch/embedding_tensor/Embedding_dataset{iteration}.pt')
            tensor_list.clear()
            print(num)
        print("over")


