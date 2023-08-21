import os
from random import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.facenet import Facenet
from nets.facenet_training import (get_lr_scheduler, set_optimizer_lr,
                                   triplet_loss, weights_init, watermark_loss)
from utils.callback import LossHistory
from utils.dataloader import FacenetDataset, LFWDataset, dataset_collate
from utils.utils import get_num_classes, show_config
from utils.utils_fit import fit_one_epoch
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import argparse

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":

    parse =  argparse.ArgumentParser()
    parse.add_argument('--seed', type=int, default=0, help='random seed')
    parse.add_argument('--input_shape', type=list, default=[160, 160, 3], help='input shape')
    parse.add_argument('--epoch', type=int, default=100, help='epoch')
    parse.add_argument('--Init_Epoch', type=int, default=10, help='Init Epoch')
    parse.add_argument('--batch_size', type=int, default=90, help='batch size')
    parse.add_argument('--num_workers', type=int, default=4, help='num workers')
    parse.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parse.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parse.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parse.add_argument('--optimizer_type', type=str, default='adam', help='optimizer type')
    parse.add_argument('--scheduler_type', type=str, default='cosin', help='scheduler type')

    parse.add_argument('--backbone', type=str, default='mobilenet', help='backbone')
    parse.add_argument('--pretrained', type=bool, default=False, help='pretrained')
    parse.add_argument('--model_path', type=str, default='', help='load model from checkpoint')

    parse.add_argument('--watermark_size', type=int, default=64, help='watermark size')
    parse.add_argument('--annotation_path', type=str, default='cls_train.txt', help='annotation path')
    parse.add_argument('--save_dir', type=str, default='logs', help='save dir')
    parse.add_argument('--save_period', type=int, default=1, help='save period')
    
    parse.add_argument('--lfw_dir_path', type=str, default='/home/zx/public/dataset/lfw', help='lfw dir path')
    parse.add_argument('--lfw_pairs_path', type=str, default='model_data/lfw_pair.txt', help='lfw pairs path')
    

    args = parse.parse_args()


    #---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存、需要pytorch1.7.1以上
    #---------------------------------------------------------------------#
    fp16            = False
    #--------------------------------------------------------#
    #   指向根目录下的cls_train.txt，读取人脸路径与标签
    #--------------------------------------------------------#
    annotation_path = args.annotation_path 
    #--------------------------------------------------------#
    #   输入图像大小，常用设置如[112, 112, 3]
    #--------------------------------------------------------#
    input_shape     = args.input_shape
    #--------------------------------------------------------#
    #   主干特征提取网络的选择
    #   mobilenet
    #   inception_resnetv1
    #--------------------------------------------------------#
    backbone        = args.backbone
    #----------------------------------------------------------------------------------------------------------------------------#
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
    #----------------------------------------------------------------------------------------------------------------------------#  
    # model_path      = "/home/lsf/facenet-pytorch/logs/ep002-loss5.022-val_loss4.791.pth"
    model_path      = args.model_path
    #----------------------------------------------------------------------------------------------------------------------------#
    #   是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
    #   如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
    #   如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
    #   如果不设置model_path，pretrained = False，此时从0开始训练。
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained      = args.pretrained

    #----------------------------------------------------------------------------------------------------------------------------#
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
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------#
    #   训练参数
    #   Init_Epoch      模型当前开始的训练世代
    #   batch_size      每次输入的图片数量
    #                   受到数据加载方式与triplet loss的影响
    #                   batch_size需要为3的倍数
    #   Epoch           模型总共训练的epochss
    #------------------------------------------------------#
    batch_size      = args.batch_size
    Init_Epoch      = args.Init_Epoch
    Epoch           = args.epoch

    #------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    #------------------------------------------------------------------#
    Init_lr             = args.lr
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    #------------------------------------------------------------------#
    optimizer_type      = args.optimizer_type
    scheduler_type      = args.scheduler_type
    momentum            = args.momentum
    weight_decay        = args.weight_decay
    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
    #------------------------------------------------------------------#
    lr_decay_type       = "cos"
    #------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值，默认每个世代都保存
    #------------------------------------------------------------------#
    save_period         = args.save_period
    #------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    #------------------------------------------------------------------#
    save_dir            = args.save_dir
    #------------------------------------------------------------------#
    #   用于设置是否使用多线程读取数据
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   内存较小的电脑可以设置为2或者0  
    #------------------------------------------------------------------#
    num_workers     = args.num_workers
    #------------------------------------------------------------------#
    #   是否开启LFW评估
    #------------------------------------------------------------------#
    # lfw_eval_flag   = True
    #------------------------------------------------------------------#
    #   LFW评估数据集的文件路径和对应的txt文件
    #------------------------------------------------------------------#
    lfw_dir_path    = args.lfw_dir_path
    lfw_pairs_path  = args.lfw_pairs_path


    num_classes = get_num_classes(annotation_path)

    watermark_size=args.watermark_size
    model = Facenet(backbone=backbone, num_classes=num_classes, pretrained=pretrained,watermark_size=watermark_size,dropout_keep_prob=0.5)

    loss            = triplet_loss()
    loss2           = watermark_loss()

    #-------------------------------------------------------#
    #   0.01用于验证，0.99用于训练
    #-------------------------------------------------------#
    val_split = 0.01
    with open(annotation_path,"r") as f:
        lines = f.readlines()
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    show_config(
        num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
        Init_Epoch = Init_Epoch, Epoch = Epoch, batch_size = batch_size, \
        Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
        save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val,watermark_size =watermark_size
    )

    if batch_size % 3 != 0:
        raise ValueError("Batch_size must be the multiple of 3.")

    optimizer = {
        'adam'  : optim.Adam(model.parameters(), 1e-3, betas = (momentum, 0.999), weight_decay = weight_decay),
        'sgd'   : optim.SGD(model.parameters(), 1e-3, momentum=momentum, nesterov=True, weight_decay = weight_decay),
        'adamw' :optim.AdamW(model.parameters(),1e-3,betas=(momentum,0.999),weight_decay=weight_decay)
    }[optimizer_type]
    
    scheduler = {
        'step'  : optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95), 
        'cosin' : optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=Epoch, eta_min=3e-4),
     }[scheduler_type]


    train_dataset   = FacenetDataset(input_shape, lines[:num_train], num_classes, random = True)
    val_dataset     = FacenetDataset(input_shape, lines[num_train:], num_classes, random = False)

    train_sampler   = None
    val_sampler     = None
    shuffle         = True
    
    gen             = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size//3, num_workers=num_workers, pin_memory=True,
                            drop_last=True, collate_fn=dataset_collate)
    gen_val         = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size//3, num_workers=num_workers, pin_memory=True,
                            drop_last=True, collate_fn=dataset_collate)
    
    
    
    
    from nets.facenet import LitFaceNet
    import pytorch_lightning as pl
    from pytorch_lightning import loggers as pl_loggers
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

        
        
    tb_logger = pl_loggers.TensorBoardLogger(save_dir)
    trainer = pl.Trainer(
                    accelerator='gpu',
                     devices=2,
                    # max_epochs=Epoch,
                    # gpus = [0,1],
                    max_epochs=1,
                    logger=tb_logger,
                    val_check_interval=0.2,
                    log_every_n_steps=50,
                    # log_every_n_steps=10, # for tiny imagenet training
                    benchmark=True, # test peroformance
                    callbacks=[
                        ModelCheckpoint(
                            # save_top_k = -1,
                            dirpath=save_dir,
                            filename='{epoch:03d}-{val_face_accuracy:.4f}-{val_wm_accuracy:.4f}',
                            every_n_train_steps=500,
                            # every_n_epochs = 10,
                            # save_last=True,
                            # save_on_train_epoch_end=True
                        )
                    ])
    model = LitFaceNet(model=model, batch_size=batch_size//3, loss=loss, loss2=loss2, optimizer=optimizer, scheduler=scheduler, watermark_size=watermark_size)

    # if len(os.listdir(save_dir)) > 0:
    #     import glob
    #     trainer.fit(model, 
    #             gen,
    #             gen_val,
    #             ckpt_path=glob.glob(save_dir+'/*.ckpt')[-1],
    #     )
    # else:
    trainer.fit(model, 
            gen,
            gen_val,
    )

    LFW_loader = torch.utils.data.DataLoader(
        LFWDataset(dir=lfw_dir_path, pairs_path=lfw_pairs_path, image_size=input_shape), batch_size=32, shuffle=False) 

