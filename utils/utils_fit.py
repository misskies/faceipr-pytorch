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


def fit_one_epoch(model_train, model, loss_history, loss, loss2, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, cuda, test_loader, Batch_size, lfw_eval_flag, fp16, scaler, save_period, save_dir,
                  local_rank, watermark_size,loss_baseline, loss_baseline_watermark_in, loss_baseline_lambda):
    total_triple_loss = 0
    total_CE_loss = 0
    total_accuracy = 0
    total_wm_accuracy = 0
    total_watermark_loss = 0

    val_total_triple_loss = 0
    val_total_CE_loss = 0
    val_total_accuracy = 0
    val_total_wm_accuracy = 0
    val_total_watermark_loss = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, labels = batch
        watermark = torch.empty(Batch_size * 3, watermark_size).uniform_(0, 1)
        watermark_in = torch.bernoulli(watermark)
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                labels = labels.cuda(local_rank)
                watermark_in = watermark_in.cuda(local_rank)
                
                if loss_baseline_watermark_in is not None:
                    loss_baseline_watermark_in = loss_baseline_watermark_in.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:

            if loss_baseline:
                outputs1, outputs2 = model_train(images, mode="train") # 128 demension feature and cls
                # transfer loss_baseline_watermark to [-1,1] 
                scaled_loss_baseline_watermark_in = loss_baseline_watermark_in * 2 -1
                # lambda_watermark = 1
                _watermark_loss =  nn.MSELoss()(outputs1, scaled_loss_baseline_watermark_in) *loss_baseline_lambda
            else:

                outputs1, outputs2, outputs3 = model_train(images, watermark_in, "train")
                _watermark_loss = loss2(outputs3, watermark_in)
            _triplet_loss = loss(outputs1, Batch_size)
            _CE_loss = nn.NLLLoss()(F.log_softmax(outputs2, dim=-1), labels)
            _loss = _triplet_loss + _CE_loss + _watermark_loss

            _loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs1, outputs2, outputs3 = model_train(images, watermark_in, "train")

                _watermark_loss = loss2(outputs3, watermark_in)
                _triplet_loss = loss(outputs1, Batch_size)
                _CE_loss = nn.NLLLoss()(F.log_softmax(outputs2, dim=-1), labels)
                _loss = _triplet_loss + _CE_loss + _watermark_loss
            # ----------------------#
            #   反向传播
            # ----------------------#
            scaler.scale(_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            if loss_baseline:
                y=extract_binary_watermark(outputs1)
            else:
                m = torch.nn.Sigmoid()
                y = m(outputs3)
                zero = torch.zeros_like(y)
                one = torch.ones_like(y)
                y = torch.where(y >= 0.5, one, y)
                y = torch.where(y < 0.5, zero, y)

            if loss_baseline:
                wm_accuracy = torch.mean((y == loss_baseline_watermark_in).type(torch.FloatTensor))
            else:
                wm_accuracy = torch.mean((y == watermark_in).type(torch.FloatTensor))
            accuracy = torch.mean((torch.argmax(F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))
        total_watermark_loss += _watermark_loss.item()
        total_triple_loss += _triplet_loss.item()
        total_CE_loss += _CE_loss.item()
        total_accuracy += accuracy.item()
        total_wm_accuracy += wm_accuracy.item()
        if local_rank == 0:
            pbar.set_postfix(**{'triple_loss': total_triple_loss / (iteration + 1),
                                'CE_loss': total_CE_loss / (iteration + 1),
                                'accuracy': total_accuracy / (iteration + 1),
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
        images, labels = batch
        watermark = torch.empty(Batch_size * 3, watermark_size).uniform_(0, 1)
        watermark_in = torch.bernoulli(watermark)
        print(images.shape)
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                labels = labels.cuda(local_rank)
                watermark_in = watermark_in.cuda(local_rank)

            optimizer.zero_grad()
            if loss_baseline:
                outputs1, outputs2 = model_train(images, mode="train") # 128 demension feature and cls
                # transfer loss_baseline_watermark to [-1,1] 
                scaled_loss_baseline_watermark_in = loss_baseline_watermark_in * 2 -1
                # lambda_watermark = 1
                _watermark_loss =  nn.MSELoss()(outputs1, scaled_loss_baseline_watermark_in) * loss_baseline_lambda
            else:
                outputs1, outputs2, outputs3 = model_train(images, watermark_in, "train")
                _watermark_loss = loss2(outputs3, watermark_in)
            _triplet_loss = loss(outputs1, Batch_size)
            _CE_loss = nn.NLLLoss()(F.log_softmax(outputs2, dim=-1), labels)
            _loss = _triplet_loss + _CE_loss + _watermark_loss

            if loss_baseline:
                y=extract_binary_watermark(outputs1)
            else:
                m = torch.nn.Sigmoid()
                y = m(outputs3)
                zero = torch.zeros_like(y)
                one = torch.ones_like(y)
                y = torch.where(y >= 0.5, one, y)
                y = torch.where(y < 0.5, zero, y)
                
            if loss_baseline:
                wm_accuracy = torch.mean((y == loss_baseline_watermark_in).type(torch.FloatTensor))
            else:
                wm_accuracy = torch.mean((y == watermark_in).type(torch.FloatTensor))
            # wm_accuracy = torch.mean((y == watermark_in).type(torch.FloatTensor))
            accuracy = torch.mean((torch.argmax(F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))

            val_total_watermark_loss += _watermark_loss.item()
            val_total_triple_loss += _triplet_loss.item()
            val_total_CE_loss += _CE_loss.item()
            val_total_accuracy += accuracy.item()
            val_total_wm_accuracy += wm_accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{'triple_loss': val_total_triple_loss / (iteration + 1),
                                'CE_loss': val_total_CE_loss / (iteration + 1),
                                'accuracy': val_total_accuracy / (iteration + 1),
                                'wm_loss': val_total_watermark_loss / (iteration + 1),
                                'wm_accuracy': val_total_wm_accuracy / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if lfw_eval_flag:
        print("开始进行LFW数据集的验证。")
        labels, distances = [], []
        for _, (data_a, data_p, label) in enumerate(test_loader):
            with torch.no_grad():
                data_a, data_p = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor)
                watermark_a = torch.empty(data_a.shape[0], watermark_size).uniform_(0, 1)
                watermark_ain = torch.bernoulli(watermark_a)
                watermark_p = torch.empty(data_p.shape[0], watermark_size).uniform_(0, 1)
                watermark_pin = torch.bernoulli(watermark_p)
                if cuda:
                    data_a, data_p = data_a.cuda(local_rank), data_p.cuda(local_rank)
                    watermark_ain = watermark_ain.cuda(local_rank)
                    watermark_pin = watermark_pin.cuda(local_rank)
                out_a, out_wa = model_train(data_a, watermark_ain)
                out_p, out_wp = model_train(data_p, watermark_pin)
                dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
            distances.append(dists.data.cpu().numpy())
            labels.append(label.data.cpu().numpy())

        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for dist in distances for subdist in dist])
        _, _, accuracy, _, _, _, _ = evaluate(distances, labels)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')

        if lfw_eval_flag:
            print('LFW_Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))

        loss_history.append_loss(epoch, np.mean(accuracy) if lfw_eval_flag else total_accuracy / epoch_step, \
                                 (total_triple_loss + total_CE_loss + total_watermark_loss) / epoch_step, (
                                             val_total_triple_loss + val_total_CE_loss + val_total_watermark_loss) / epoch_step_val,
                                 val_total_wm_accuracy / epoch_step_val, val_total_watermark_loss / epoch_step_val,
                                 total_accuracy / epoch_step ,val_total_accuracy / epoch_step_val)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))#
        print('Total Loss: %.4f' % ((total_triple_loss + total_CE_loss + total_watermark_loss) / epoch_step))
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            filename =  'ep%03d-loss%.3f-val_loss%.3f.pth' % ((epoch + 1), 
                                                             (total_triple_loss + total_CE_loss + total_watermark_loss) / epoch_step, 
                                                             (val_total_triple_loss + val_total_CE_loss + val_total_watermark_loss) / epoch_step_val)

            loss_history.save_model(model, filename)
            # torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % ((epoch + 1),
            #                                                                                             (
            #                                                                                                         total_triple_loss + total_CE_loss + total_watermark_loss) / epoch_step,
            #                                                                                             (
            #                                                                                                         val_total_triple_loss + val_total_CE_loss + val_total_watermark_loss) / epoch_step_val)))