import datetime
import os
#
import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import yaml

class LossHistory():
    def __init__(self, log_dir, model, input_shape,watermark_size,PostNet):
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        self.log_dir    = os.path.join(log_dir, "loss_" + str(time_str))
        self.acc        = []
        self.losses     = []
        self.val_loss   = []
        self.face_train_acc = []
        self.face_val_acc =  []
        self.wm_acc = []
        #
        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        a = torch.empty(2, watermark_size).uniform_(0, 1)
        a=torch.bernoulli(a)
        if PostNet :
            watermark = torch.empty(2, watermark_size).uniform_(0, 1)
            watermark_in = torch.bernoulli(watermark)
            dummy_input = torch.randn(2,128),watermark_in
        else:
            dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1]),a
        #dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1]),None,"origin"
        self.writer.add_graph(model, dummy_input)

    def append_loss(self, epoch, acc, loss, val_loss,wm_acc,wm_loss,face_train_acc,face_val_acc):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.wm_acc.append(wm_acc)
        self.face_train_acc.append((face_train_acc))
        self.face_val_acc.append(face_val_acc)
        self.acc.append(acc)
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        with open(os.path.join(self.log_dir, "epoch_face_train_acc.txt"), 'a') as f:
            f.write(str(face_train_acc))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_face_val_acc.txt"), 'a') as f:
            f.write(str(face_val_acc))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_wmloss.txt"), 'a') as f:
            f.write(str(wm_loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_wmacc.txt"), 'a') as f:
            f.write(str(wm_acc))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_acc.txt"), 'a') as f:
            f.write(str(acc))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()
        self.acc_plot()

    def acc_plot(self):
        iters = range(len(self.face_val_acc))

        plt.figure()
        plt.plot(iters, self.face_train_acc, 'red', linewidth = 2, label='face_train_acc')
        plt.plot(iters, self.face_val_acc, 'coral', linewidth = 2, label='face_val_acc')
        plt.plot(iters, self.wm_acc, 'blue', linewidth = 2, label='wm_acc')
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "Train_accuracy.png"))
        plt.cla()
        plt.close("all")

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--',
                     linewidth=2, label='smooth val loss')
        except:
            pass
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla()
        plt.close("all")

        plt.figure()
        plt.plot(iters, self.acc, 'red', linewidth=2, label='lfw acc')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            plt.plot(iters, scipy.signal.savgol_filter(self.acc, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth lfw acc')
        except:
            pass
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Lfw Acc')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_acc.png"))
        plt.cla()
        plt.close("all")
    
    def save_args(self, args, filename="args.yaml"):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        with open(os.path.join(self.log_dir, filename), 'w') as file:
            yaml.dump(vars(args), file)

    def save_tensor(self, tensor, filename="tensor.pt"):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        torch.save(tensor, os.path.join(self.log_dir, filename)) 
    
    def save_model(self, model, filename="model.pth"):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        torch.save(model.state_dict(), os.path.join(self.log_dir, filename))