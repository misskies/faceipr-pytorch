import os
from random import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from nets.PostNet import Postnet
from nets.facenet import Facenet, Facenet_loss, Facenet_128
from nets.facenet_training import (get_lr_scheduler, set_optimizer_lr,
                                   triplet_loss, weights_init, watermark_loss)
from utils.callback import LossHistory
from utils.dataloader import FacenetDataset, LFWDataset, dataset_collate, EmbeddingDataset
from utils.utils import get_num_classes, show_config
from utils.utils_PostNet_fit import PostNet_fit_one_epoch
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
    
    parse =  argparse.ArgumentParser()
    parse.add_argument('--seed', type=int, default=20, help='random seed')
    parse.add_argument('--input_shape', type=list, default=[160, 160, 3], help='input shape')
    parse.add_argument('--epoch', type=int, default=100, help='epoch')
    parse.add_argument('--Init_Epoch', type=int, default=0, help='Init Epoch')
    parse.add_argument('--batch_size', type=int, default=150, help='batch size')
    parse.add_argument('--num_workers', type=int, default=4, help='num workers')
    parse.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parse.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parse.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parse.add_argument('--optimizer_type', type=str, default='adam', help='optimizer type')
    # parse.add_argument('--scheduler_type', type=str, default='cosin', help='scheduler type')

    parse.add_argument('--backbone', type=str, default='mobilenet', help='backbone')
    parse.add_argument('--pretrained', type=bool, default=False, help='pretrained')
    parse.add_argument('--model_path', type=str, default='/home/lsf/facenet-pytorch/logs/ep002-loss5.022-val_loss4.791.pth', help='load model from checkpoint')
    parse.add_argument('--decoder_arch', type=str, default='fc', choices=['fc', 'conv', 'attn', 'conv_attn'],help='The architecuture of decoder')

    parse.add_argument('--watermark_size', type=int, default=32, help='watermark size')
    parse.add_argument('--annotation_path', type=str, default='cls_train.txt', help='annotation path')
    parse.add_argument('--save_dir', type=str, default='logs', help='save dir')
    parse.add_argument('--save_period', type=int, default=1, help='save period')
    
    parse.add_argument('--lfw_dir_path', type=str, default='lfw', help='lfw dir path')
    parse.add_argument('--lfw_pairs_path', type=str, default='model_data/lfw_pair.txt', help='lfw pairs path')

    parse.add_argument('--original', type=bool, default=False, help='Whether modulate the mode')

    parse.add_argument('--loss_baseline', type=bool, default=False, help='Whether train loss_baseline')
    parse.add_argument('--loss_baseline_lambda', type=float, default=1.0, help='the lambda watermark value in loss baseline')

    parse.add_argument('--robustness', type=str, default='none', help='',  choices=['none', 'noise', 'flip','round','random_del','combine'])
    parse.add_argument('--noise_power',type=float,default=0.1,help='Noise injection power')

    parse.add_argument('--embed_128', default=False,  action="store_true", help='Whether use 128 demension embedding')
    
    parse.add_argument('--local_rank', type=int, default=0, help='local rank')

    parse.add_argument('--PostNet', type=bool, default=False, help='Whether train PostNet')

    args = parse.parse_args()

    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    Cuda            = True
    #---------------------------------------------------------------------#
    #   distributed     用于指定是否使用单机多卡分布式运行
    #                   终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡。
    #                   Windows系统下默认使用DP模式调用所有显卡，不支持DDP。
    #   DP模式：
    #       设置            distributed = False
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP模式：
    #       设置            distributed = True
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   sync_bn     是否使用sync_bn，DDP模式多卡可用
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存、需要pytorch1.7.1以上
    #---------------------------------------------------------------------#
    fp16            = False
    #--------------------------------------------------------#
    #   是否只训练原始任务(是否加入Encoder-Decoder,以及调制卷积层)
    #--------------------------------------------------------#
    original = args.original
    #--------------------------------------------------------#
    #   是否训练loss_function_baseline
    #--------------------------------------------------------#
    loss_baseline=args.loss_baseline
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
    lfw_eval_flag   = True
    #------------------------------------------------------------------#
    #   LFW评估数据集的文件路径和对应的txt文件
    #------------------------------------------------------------------#
    lfw_dir_path    = args.lfw_dir_path
    lfw_pairs_path  = args.lfw_pairs_path
    #------------------------------------------------------#
    #   设置用到的显卡
    #------------------------------------------------------#
    PostNet = args.PostNet
    #------------------------------------------------------#
    #   设置用到的显卡
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    print(ngpus_per_node)
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    num_classes = get_num_classes(annotation_path)
    #---------------------------------#
    #   载入模型并加载预训练权重
    #---------------------------------#
    # init_type="kaiming"
    watermark_size=args.watermark_size
    
    
    loss_baseline_watermark_in = None
    if loss_baseline :
        model = Facenet_loss(backbone=backbone, num_classes=num_classes, pretrained=pretrained,
                        dropout_keep_prob=0.5)
        # watermark_size=1024
        watermark_size=128
        
        watermark = torch.empty(1, watermark_size).uniform_(0, 1)
        loss_baseline_watermark_in = torch.bernoulli(watermark).repeat(batch_size, 1)
        # torch.save(loss_baseline_watermark_in, os.path.join(save_dir, 'loss_baseline_watermark_in.pt'))

    elif args.embed_128:

        model = Facenet_128(backbone=backbone, num_classes=num_classes, pretrained=pretrained,
                    watermark_size=watermark_size,dropout_keep_prob=0.5, robustness=args.robustness,noise_power=args.noise_power)
    elif PostNet:
        model = Postnet(watermark_size=watermark_size)
    else:
        model = Facenet(backbone=backbone, num_classes=num_classes, pretrained=pretrained,
                    watermark_size=watermark_size,dropout_keep_prob=0.5, robustness=args.robustness,noise_power=args.noise_power, decoder_arch=args.decoder_arch)
    #weights_init(model,init_type)
    if model_path != '':
        #------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        #------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                if ("Encoder" in k or "Decoder" in k) and original :
                    no_load_key.append(k)
                else:
                    temp_dict[k] = v
                    load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   显示没有匹配上的Key
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    loss            = triplet_loss()
    loss2           = watermark_loss()
    #----------------------#
    #   记录Loss
    #----------------------#
    if local_rank == 0:
        loss_history = LossHistory(save_dir, model, input_shape=input_shape,watermark_size=watermark_size,PostNet=PostNet)
    else:
        loss_history = None
        
    #------------------------------------------------------------------#
    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #----------------------------#
    #   多卡同步Bn
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            #----------------------------#
            #   多卡平行运行
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    #---------------------------------#
    #   LFW估计
    #---------------------------------#
    LFW_loader = torch.utils.data.DataLoader(
        LFWDataset(dir=lfw_dir_path, pairs_path=lfw_pairs_path, image_size=input_shape), batch_size=32, shuffle=False) if lfw_eval_flag else None

    #-------------------------------------------------------#
    #   0.01用于验证，0.99用于训练
    #-------------------------------------------------------#
    val_split = 0.01
    with open(annotation_path,"r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    setup_seed(args.seed)
    show_config(
        num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
        Init_Epoch = Init_Epoch, Epoch = Epoch, batch_size = batch_size, \
        Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
        save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val,watermark_size =watermark_size
    )
    loss_history.save_args(args, "args.yaml")
    if loss_baseline:
        loss_history.save_tensor(loss_baseline_watermark_in, "loss_baseline_watermark_in.pt")

    if True:
        if batch_size % 3 != 0:
            raise ValueError("Batch_size must be the multiple of 3.")
        #-------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        #-------------------------------------------------------------------#
        nbs             = 64
        lr_limit_max    = (1e-3) if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        print(Init_lr_fit)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        print(Min_lr_fit)

        #---------------------------------------#
        #   根据optimizer_type选择优化器
        #---------------------------------------#
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay = weight_decay),
            'adamw' :optim.AdamW(model.parameters(),Init_lr_fit,betas=(momentum,0.999),weight_decay=weight_decay)
        }[optimizer_type]

        #---------------------------------------#
        #   获得学习率下降的公式
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)
        
        #---------------------------------------#
        #   判断每一个世代的长度
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        #---------------------------------------#
        #   构建数据集加载器。
        #---------------------------------------#
        if PostNet :
            file_path = "/home/lsf/facenet-pytorch/embedding_tensor"
            dataset = EmbeddingDataset(file_path)
            train_size = int(len(dataset)*(1-val_split))
            val_size = len(dataset) - train_size
            train_dataset,val_dataset =random_split(dataset,[train_size,val_size])
        else:
            train_dataset   = FacenetDataset(input_shape, lines[:num_train], num_classes, random = True)
            val_dataset     = FacenetDataset(input_shape, lines[num_train:], num_classes, random = False)

        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        if PostNet :
            gen = DataLoader(train_dataset,shuffle=shuffle,batch_size=batch_size)
            gen_val = DataLoader(val_dataset,shuffle=shuffle,batch_size=batch_size)
        else:
            gen             = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size//3, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate, sampler=train_sampler)
            gen_val         = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size//3, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate, sampler=val_sampler)

        for epoch in range(Init_Epoch, Epoch):
            if distributed:
                train_sampler.set_epoch(epoch)##
                
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            if original :
                original_fit_one_epoch(model_train, model, loss_history, loss, loss2, optimizer, epoch, epoch_step,
                              epoch_step_val, gen, gen_val, Epoch, Cuda, LFW_loader, batch_size // 3, lfw_eval_flag,
                              fp16, scaler, save_period, save_dir, local_rank, watermark_size)
            elif PostNet:
                PostNet_fit_one_epoch(model_train, model, loss_history, loss,loss2,optimizer, epoch, epoch_step,
                              epoch_step_val, gen,gen_val, Epoch, Cuda, LFW_loader, batch_size, lfw_eval_flag,
                              fp16, scaler, save_period, save_dir, local_rank,watermark_size)
            else:
                fit_one_epoch(model_train, model, loss_history, loss,loss2,optimizer, epoch, epoch_step,
                              epoch_step_val, gen,gen_val, Epoch, Cuda, LFW_loader, batch_size//3, lfw_eval_flag,
                              fp16, scaler, save_period, save_dir, local_rank,watermark_size,loss_baseline, loss_baseline_watermark_in, args.loss_baseline_lambda)

        if local_rank == 0:
            loss_history.writer.close()
