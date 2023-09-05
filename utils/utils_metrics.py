import numpy as np
import torch
from scipy import interpolate
from sklearn.model_selection import KFold
from tqdm import tqdm

from nets.baseline import extract_binary_watermark


def evaluate(distances, labels, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    tpr, fpr, accuracy, best_thresholds = calculate_roc(thresholds, distances,
        labels, nrof_folds=nrof_folds)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds, distances,
        labels, 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far, best_thresholds 

def calculate_roc(thresholds, distances, labels, nrof_folds=10):

    nrof_pairs = min(len(labels), len(distances))
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, distances[train_set], labels[train_set])

        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, distances[test_set], labels[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], distances[test_set], labels[test_set])
        tpr = np.mean(tprs,0)
        fpr = np.mean(fprs,0)
    return tpr, fpr, accuracy, thresholds[best_threshold_index]

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc

def calculate_val(thresholds, distances, labels, far_target=1e-3, nrof_folds=10):
    nrof_pairs = min(len(labels), len(distances))
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, distances[train_set], labels[train_set])
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, distances[test_set], labels[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean

def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    if n_diff == 0:
        n_diff = 1
    if n_same == 0:#
        return 0,0
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

def test(test_loader, model, png_save_path, log_interval, batch_size, cuda,watermark_size,origin,robustness,noise_power,test_robustness,PostNet,model1=None):
    labels, distances,sameface_distances= [], [],[]
    wm_accuracy=0
    acc_wm=0
    num=0
    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        with torch.no_grad():
            watermark = torch.empty(data_a.shape[0], watermark_size).uniform_(0, 1)
            watermark_in = torch.bernoulli(watermark)
            watermark_in1 = torch.bernoulli(watermark)
            #--------------------------------------#
            #   加载数据，设置成cuda
            #--------------------------------------#
            data_a, data_p      = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor)
            if cuda:
                data_a, data_p,data_wm,data_wm1= data_a.cuda(), data_p.cuda(),watermark_in.cuda(),watermark_in1.cuda()
            #--------------------------------------#
            #   传入模型预测，获得预测结果
            #   获得预测结果的距离
            #--------------------------------------#
            if origin:
                out_a,out_wm1        =  model(data_a,data_wm,"Unmd_predict", robustness=robustness, noise_power=noise_power)
                out_p,out_wm2        =  model(data_p,data_wm,"Unmd_predict", robustness='none')
                dists                = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
                sameface_dists  = dists
            else:
                if PostNet :
                    model1_wm = None
                    out_a = model1(data_a, model1_wm,"original_predict")
                    out_a1 = model1(data_a,model1_wm,"original_predict")
                    out_p = model1(data_p, model1_wm,"original_predict")
                    out_a, out_wm1 = model(out_a, data_wm)
                    out_a1,out_tmp = model(out_a1,data_wm1)
                    out_p, out_wm2 = model(out_p, data_wm)
                    sameface_dists       = dists # avoid error
                else:
                    out_a, out_wm1 = model(data_a, data_wm)
                    out_a1,out_tmp = model(data_a,data_wm1)
                    out_p, out_wm2 = model(data_p, data_wm,"no-noise_predict")
                dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
                sameface_dists=torch.sqrt(torch.sum((out_a1 - out_p) ** 2, 1))
                m = torch.nn.Sigmoid()
                y1 = m(out_wm1)
                y2 = m(out_wm2)
                zero = torch.zeros_like(y1)
                one = torch.ones_like(y1)
                y1 = torch.where(y1 >= 0.5, one, y1)
                y1 = torch.where(y1 < 0.5, zero, y1)
                y2 = torch.where(y2 >= 0.5, one, y2)
                y2 = torch.where(y2 < 0.5, zero, y2)
                acc_wm += torch.mean((y1 == data_wm).type(torch.FloatTensor))
                #acc_wm += torch.mean((y2 == data_wm).type(torch.FloatTensor))
                num += 1
        #--------------------------------------#
        #   将结果添加进列表中
        #--------------------------------------#
        distances.append(dists.data.cpu().numpy())
        sameface_distances.append(sameface_dists.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())

        #--------------------------------------#
        #   打印
        #--------------------------------------#
        if batch_idx % log_interval == 0:
            pbar.set_description('Test Epoch: [{}/{} ({:.0f}%)]'.format(
                batch_idx * batch_size, len(test_loader.dataset),
                100. * batch_idx / len(test_loader)))

    #--------------------------------------#
    #   转换成numpy
    #--------------------------------------#
    labels      = np.array([sublabel for label in labels for sublabel in label])
    distances   = np.array([subdist for dist in distances for subdist in dist])
    sameface_distances= np.array([subdist for dist in sameface_distances for subdist in dist])
    if not origin :
        acc_wm/=num
    tpr, fpr, accuracy, val, val_std, far, best_thresholds = evaluate(distances,labels)
    print()
    print('WatermarkAccuracy: %2.5f' %(acc_wm))
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Best_thresholds: %2.5f' % best_thresholds)
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    plot_roc(fpr, tpr, figure_name = png_save_path)

    print("Same face different Watermark:")
    tpr, fpr, accuracy, val, val_std, far, best_thresholds = evaluate(sameface_distances,labels)
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Best_thresholds: %2.5f' % best_thresholds)
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    #
    #
    if origin:
        with open(f"eval_robustness/origin_{test_robustness}_LFWacc.txt", 'a') as f:
            f.write(str(np.mean(accuracy)))
            f.write("\n")
    else:

        with open(f"eval_robustness/wm32_{test_robustness}_LFWacc.txt", 'a') as f:
            f.write(str(np.mean(accuracy)))
            f.write("\n")
        with open(f"eval_robustness/wm32_{test_robustness}_wmacc.txt", 'a') as f:
            f.write(str(acc_wm.item()))
            f.write("\n")

def loss_baseline_test(test_loader, model, png_save_path, log_interval, batch_size, cuda,watermark_size,robustness,noise_power,test_robustness, wm_path):
    labels, distances,sameface_distances= [], [],[]
    wm_accuracy=0
    acc_wm=0
    num=0
    pbar = tqdm(enumerate(test_loader))
    # path = "/home/lsf/public/collaboration/facenet-pytorch/facenet-pytorch/trained_weight/faceweb_lossEmbed_mobilenet/loss_baseline_watermark_in.pt"
    path = wm_path 
    dict = torch.load(path)
    dict = list(dict)
    for batch_idx, (data_a, data_p, label) in pbar:
        with torch.no_grad():
            watermark_in = dict[0].repeat(data_a.shape[0], 1)
            #--------------------------------------#
            #   加载数据，设置成cuda
            #--------------------------------------#
            data_a, data_p      = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor)
            if cuda:
                data_a, data_p,data_wm= data_a.cuda(), data_p.cuda(),watermark_in.cuda()
            #--------------------------------------#
            #   传入模型预测，获得预测结果
            #   获得预测结果的距离
            #--------------------------------------#
            out_a, out_wm1 = model(data_a, data_wm)
            out_p, out_wm2 = model(data_p, data_wm,"no-noise_predict")
            dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
            y1 = extract_binary_watermark(out_a)
            y2 = extract_binary_watermark(out_p)
            acc_wm += torch.mean((y1 == data_wm).type(torch.FloatTensor))
            #acc_wm += torch.mean((y2 == data_wm).type(torch.FloatTensor))
            num += 1

        #--------------------------------------#
        #   将结果添加进列表中
        #--------------------------------------#
        distances.append(dists.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())

        #--------------------------------------#
        #   打印
        #--------------------------------------#
        if batch_idx % log_interval == 0:
            pbar.set_description('Test Epoch: [{}/{} ({:.0f}%)]'.format(
                batch_idx * batch_size, len(test_loader.dataset),
                100. * batch_idx / len(test_loader)))

    #--------------------------------------#
    #   转换成numpy
    #--------------------------------------#
    labels      = np.array([sublabel for label in labels for sublabel in label])
    distances   = np.array([subdist for dist in distances for subdist in dist])
    sameface_distances= np.array([subdist for dist in sameface_distances for subdist in dist])
    acc_wm/=num
    tpr, fpr, accuracy, val, val_std, far, best_thresholds = evaluate(distances,labels)
    print()
    print('WatermarkAccuracy: %2.5f' %(acc_wm))
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Best_thresholds: %2.5f' % best_thresholds)
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    plot_roc(fpr, tpr, figure_name = png_save_path)

    with open(f"eval_robustness/loss-baseline_{test_robustness}_LFWacc.txt", 'a') as f:
        f.write(str(np.mean(accuracy)))
        f.write("\n")
    with open(f"eval_robustness/loss-baseline_{test_robustness}_wmacc.txt", 'a') as f:
        f.write(str(acc_wm.item()))
        f.write("\n")

def LSB_test(test_loader, model, png_save_path, log_interval, batch_size, cuda,watermark_size):
    labels, distances,sameface_distances= [], [],[]
    wm_accuracy=0
    acc_wm=0
    num=0
    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        with torch.no_grad():
            watermark = torch.empty(data_a.shape[0], watermark_size).uniform_(0, 1)
            watermark_in = torch.bernoulli(watermark)
            watermark_in1 = torch.bernoulli(watermark)
            #--------------------------------------#
            #   加载数据，设置成cuda
            #--------------------------------------#
            data_a, data_p      = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor)
            if cuda:
                data_a, data_p,data_wm,data_wm1= data_a.cuda(), data_p.cuda(),watermark_in.cuda(),watermark_in1.cuda()
            #--------------------------------------#
            #   传入模型预测，获得预测结果
            #   获得预测结果的距离
            #--------------------------------------#
            out_a, out_wm1 = model(data_a, data_wm,"LSB")
            out_a1,out_tmp = model(data_a,data_wm1,"LSB")
            out_p, out_wm2 = model(data_p, data_wm,"no-noise_LSB")
            dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
            sameface_dists=torch.sqrt(torch.sum((out_a1 - out_p) ** 2, 1))
            acc_wm += torch.mean((out_wm1 == data_wm).type(torch.FloatTensor))
            #acc_wm += torch.mean((out_wm2 == data_wm).type(torch.FloatTensor))
            num += 1
        #--------------------------------------#
        #   将结果添加进列表中
        #--------------------------------------#
        distances.append(dists.data.cpu().numpy())
        sameface_distances.append(sameface_dists.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())

        #--------------------------------------#
        #   打印
        #--------------------------------------#
        if batch_idx % log_interval == 0:
            pbar.set_description('Test Epoch: [{}/{} ({:.0f}%)]'.format(
                batch_idx * batch_size, len(test_loader.dataset),
                100. * batch_idx / len(test_loader)))

    #--------------------------------------#
    #   转换成numpy
    #--------------------------------------#
    labels      = np.array([sublabel for label in labels for sublabel in label])
    distances   = np.array([subdist for dist in distances for subdist in dist])
    sameface_distances= np.array([subdist for dist in sameface_distances for subdist in dist])
    tpr, fpr, accuracy, val, val_std, far, best_thresholds = evaluate(distances,labels)
    print()
    acc_wm /= num
    print('WatermarkAccuracy: %2.5f' %(acc_wm))
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Best_thresholds: %2.5f' % best_thresholds)
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    plot_roc(fpr, tpr, figure_name = png_save_path)

    print("Same face different Watermark:")
    tpr, fpr, accuracy, val, val_std, far, best_thresholds = evaluate(sameface_distances,labels)
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Best_thresholds: %2.5f' % best_thresholds)
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

    # with open("eval_robustness/LSB_round_wmacc.txt", 'a') as f:
    #      f.write(str(acc_wm.item()))
    #      f.write("\n")



def post_test(test_loader, model, png_save_path, log_interval, batch_size, cuda,watermark_size, post_method, robustness, noise_power,test_robustness):
    labels, distances,sameface_distances= [], [],[]
    wm_accuracy=0
    acc_wm=0
    num=0
    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        with torch.no_grad():
            watermark = torch.empty(data_a.shape[0], watermark_size).uniform_(0, 1)
            watermark_in = torch.bernoulli(watermark)
            watermark_in1 = torch.bernoulli(watermark)
            #--------------------------------------#
            #   加载数据，设置成cuda
            #--------------------------------------#
            data_a, data_p      = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor)
            if cuda:
                data_a, data_p,data_wm,data_wm1= data_a.cuda(), data_p.cuda(),watermark_in.cuda(),watermark_in1.cuda()
            #--------------------------------------#
            #   传入模型预测，获得预测结果
            #   获得预测结果的距离
            #--------------------------------------#
            out_a, out_wm1 = model(data_a, data_wm, post_method, robustness, noise_power=noise_power)
            out_a1,out_tmp = model(data_a,data_wm1, post_method, robustness, noise_power=noise_power)
            out_p, out_wm2 = model(data_p, data_wm, post_method, robustness="none")
            dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
            sameface_dists=torch.sqrt(torch.sum((out_a1 - out_p) ** 2, 1))
            acc_wm += torch.mean((out_wm1 == data_wm).type(torch.FloatTensor))
            # acc_wm += torch.mean((out_wm2 == data_wm).type(torch.FloatTensor))
            num += 1
        #--------------------------------------#
        #   将结果添加进列表中
        #--------------------------------------#
        distances.append(dists.data.cpu().numpy())
        sameface_distances.append(sameface_dists.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())

        #--------------------------------------#
        #   打印
        #--------------------------------------#
        if batch_idx % log_interval == 0:
            pbar.set_description('Test Epoch: [{}/{} ({:.0f}%)]'.format(
                batch_idx * batch_size, len(test_loader.dataset),
                100. * batch_idx / len(test_loader)))

    #--------------------------------------#
    #   转换成numpy
    #--------------------------------------#
    labels      = np.array([sublabel for label in labels for sublabel in label])
    distances   = np.array([subdist for dist in distances for subdist in dist])
    sameface_distances= np.array([subdist for dist in sameface_distances for subdist in dist])
    tpr, fpr, accuracy, val, val_std, far, best_thresholds = evaluate(distances,labels)
    print()
    acc_wm /= num
    print('WatermarkAccuracy: %2.5f' %(acc_wm))
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Best_thresholds: %2.5f' % best_thresholds)
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    plot_roc(fpr, tpr, figure_name = png_save_path)

    print("Same face different Watermark:")
    tpr, fpr, accuracy, val, val_std, far, best_thresholds = evaluate(sameface_distances,labels)
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Best_thresholds: %2.5f' % best_thresholds)
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

    with open(f"eval_robustness/{post_method}_{test_robustness}_LFWacc.txt", 'a') as f:
         f.write(str(np.mean(accuracy)))
         f.write("\n")
    with open(f"eval_robustness/{post_method}_{test_robustness}_wmacc.txt", 'a') as f:
         f.write(str(acc_wm.item()))
         f.write("\n")
def plot_roc(fpr, tpr, figure_name = "roc.png"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    fig.savefig(figure_name, dpi=fig.dpi)
