

import math
from functools import partial

import numpy as np
import torch
from torch import nn, Tensor


def watermark_loss():
    def _watermark_loss(watermark_fin,watermark_in):
        ls = nn.BCEWithLogitsLoss()
        loss = ls(watermark_fin,watermark_in)
        return loss
    return _watermark_loss

def triplet_loss(alpha = 0.2):
    def _triplet_loss(y_pred,Batch_size):
        anchor, positive, negative = y_pred[:int(Batch_size)], y_pred[int(Batch_size):int(2*Batch_size)], y_pred[int(2*Batch_size):]

        eps=1e-6
        pos_dist = torch.sqrt(torch.sum(torch.pow(anchor - positive,2), axis=-1) + eps)
        neg_dist = torch.sqrt(torch.sum(torch.pow(anchor - negative,2), axis=-1) + eps)
        
        keep_all = (neg_dist - pos_dist < alpha).cpu().numpy().flatten()
        hard_triplets = np.where(keep_all == 1)

        pos_dist = pos_dist[hard_triplets]
        neg_dist = neg_dist[hard_triplets]

        basic_loss = pos_dist - neg_dist + alpha
        loss = torch.sum(basic_loss) / torch.max(torch.tensor(1), torch.tensor(len(hard_triplets[0])))
        return loss

    return _triplet_loss

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.1, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.3, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# if __name__ == '__main__':
#     watermark_fin=Tensor([36.4457, 36.7833, 3.2325, 3.2487, 3.3875, 37.0120, 36.7522, 36.4497, 3.8769,
#         37.0945, 36.7675, 36.4894, 36.6358, 3.2729, 3.6695, 36.5324, 36.7415, 3.6016,
#         36.9260, 3.6408, 3.5882, 36.1815, 36.6912, 36.7947, 3.3250, 36.4834, 3.2450,
#         36.7019, 36.3268, 3.4150, 36.4782, 3.5255])
#     watermark_in=Tensor([1., 1., 0., 0., 0., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 1., 1., 0.,
#         1., 0., 0., 1., 1., 1., 0., 1., 0., 1., 1., 0., 1., 0.])
#     ls = nn.BCEWithLogitsLoss()
#     loss = ls(watermark_fin, watermark_in)
#     print(loss)s