import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from nets.baseline import extract_binary_watermark
from utils.utils import get_lr
from utils.utils_metrics import evaluate

import torch
import torch.nn as nn


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss#
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


def PostNet_fit_one_epoch(model_train, model, loss_history, loss, loss2, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, cuda, test_loader, Batch_size, lfw_eval_flag, fp16, scaler, save_period, save_dir,
                  local_rank, watermark_size):

    total_embedding_loss = 0
    total_wm_accuracy = 0
    total_watermark_loss = 0

    val_total_embedding_loss = 0
    val_total_wm_accuracy = 0
    val_total_watermark_loss = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        embedding_in = batch
        watermark = torch.empty(batch.shape[0], watermark_size).uniform_(0, 1)
        watermark_in = torch.bernoulli(watermark)
        with torch.no_grad():
            if cuda:
                embedding_in = embedding_in.cuda(local_rank)
                watermark_in = watermark_in.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            embedding_out, watermark_out= model_train(embedding_in,watermark_in)
            _watermark_loss = loss2(watermark_out, watermark_in)
            loss = nn.MSELoss()
            _embedding_loss = loss(embedding_out,embedding_in)
            _loss = _embedding_loss + _watermark_loss

            _loss.backward()
            optimizer.step()

        with torch.no_grad():
            m = torch.nn.Sigmoid()
            y = m(watermark_out)
            zero = torch.zeros_like(y)
            one = torch.ones_like(y)
            y = torch.where(y >= 0.5, one, y)
            y = torch.where(y < 0.5, zero, y)
            wm_accuracy = torch.mean((y == watermark_in).type(torch.FloatTensor))
        total_watermark_loss += _watermark_loss.item()
        total_embedding_loss += _embedding_loss.item()
        total_wm_accuracy += wm_accuracy.item()
        if local_rank == 0:
            pbar.set_postfix(**{'embedding_loss': total_embedding_loss / (iteration + 1),
                                'wm_loss': total_watermark_loss / (iteration + 1),
                                'wm_accuracy ': total_wm_accuracy / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        embedding_in = batch
        watermark = torch.empty(batch.shape[0], watermark_size).uniform_(0, 1)
        watermark_in = torch.bernoulli(watermark)
        with torch.no_grad():
            if cuda:
                embedding_in = embedding_in.cuda(local_rank)
                watermark_in = watermark_in.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            embedding_out, watermark_out = model_train(embedding_in,watermark_in)
            _watermark_loss = loss2(watermark_out, watermark_in)
            loss = nn.MSELoss()
            _embedding_loss = loss(embedding_out, embedding_in)
            _loss = _embedding_loss + _watermark_loss

            _loss.backward()
            optimizer.step()

        with torch.no_grad():
            m = torch.nn.Sigmoid()
            y = m(watermark_out)
            zero = torch.zeros_like(y)
            one = torch.ones_like(y)
            y = torch.where(y >= 0.5, one, y)
            y = torch.where(y < 0.5, zero, y)
            wm_accuracy = torch.mean((y == watermark_in).type(torch.FloatTensor))

            val_total_watermark_loss += _watermark_loss.item()
            val_total_embedding_loss += _embedding_loss.item()
            val_total_wm_accuracy += wm_accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{'embedding_loss': val_total_embedding_loss / (iteration + 1),
                                'wm_loss': val_total_watermark_loss / (iteration + 1),
                                'wm_accuracy': val_total_wm_accuracy / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)


    if local_rank == 0:
        pbar.close()
        print('Finish Validation')


        loss_history.append_loss(epoch, 0 if lfw_eval_flag else 0, \
                                 (total_embedding_loss +  + total_watermark_loss) / epoch_step, (
                                         val_total_embedding_loss + val_total_watermark_loss) / epoch_step_val,
                                 val_total_wm_accuracy / epoch_step_val, val_total_watermark_loss / epoch_step_val,
                                 0, 0)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))  #
        print('Total Loss: %.4f' % ((total_embedding_loss + total_watermark_loss) / epoch_step))
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            filename = 'ep%03d-loss%.3f-val_loss%.3f.pth' % ((epoch + 1),
                                                             (
                                                                         total_embedding_loss  + total_watermark_loss) / epoch_step,
                                                             (
                                                                         val_total_embedding_loss + val_total_watermark_loss) / epoch_step_val)

            loss_history.save_model(model, filename)
            # torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % ((epoch + 1),
            #                                                                                             (
            #                                                                                                         total_triple_loss + total_CE_loss + total_watermark_loss) / epoch_step,
            #                                                                                             (
            #                                                                                                         val_total_triple_loss + val_total_CE_loss + val_total_watermark_loss) / epoch_step_val)))