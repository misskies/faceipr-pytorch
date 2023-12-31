# python -m torch.distributed.launch train.py --model_path="" --watermark_size=32 --annotation_path="cls_train.txt" --lfw_dir_path="/home/zx/public/dataset/lfw" --lfw_pairs_path="model_data/lfw_pair.txt"
# python -m torch.distributed.launch train.py --model_path="" --watermark_size=32 --annotation_path="cls_train.txt" --lfw_dir_path="/home/zx/public/dataset/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=9
# python -m torch.distributed.launch train.py --model_path="" --watermark_size=32 --annotation_path="cls_train.txt" --lfw_dir_path="/home/zx/public/dataset/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=9 --robustness="flip"

#CUDA_VISIBLE_DEVICES=0 python train.py --model_path="" --watermark_size=32 --annotation_path="cls_train.txt" --lfw_dir_path="lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=96

################################1. no modulation on watermark_mobilenet
# python train.py --model_path="" --annotation_path="cls_train.txt" --lfw_dir_path="/home/zx/public/dataset/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=9 --original=True

# python train.py --backbone="resnet34" --model_path="" --annotation_path="faceweb_cls_train.txt" --lfw_dir_path="datasets/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=120 --original=True --save_period=5
#
#python train.py --backbone="resnet34" --model_path="" --annotation_path="celeba_cls_train.txt" --lfw_dir_path="datasets/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=120 --original=True --save_period=5



################################2.load no modulation model + encoder + decoder
# python train.py --model_path="/home/lsf/facenet-pytorch/trained_weight/faceweb_unmd_mobilenet/ep042-loss0.410-val_loss2.268.pth" --watermark_size==32 --annotation_path="cls_train.txt" --lfw_dir_path="/home/zx/public/dataset/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=9 --epoch=10

# python train.py --model_path="/home/lsf/facenet-pytorch/trained_weight/faceweb_unmd_mobilenet/ep042-loss0.410-val_loss2.268.pth" --watermark_size==64 --annotation_path="cls_train.txt" --lfw_dir_path="/home/zx/public/dataset/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=9  --epoch=50
#

#resnet34
#python train.py --backbone="resnet34" --model_path="trained_weight/faceweb_unmd_resnet34/ep030-loss0.421-val_loss1.849.pth" --watermark_size=32 --annotation_path="faceweb_cls_train.txt" --lfw_dir_path="datasets/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=60  --epoch=20

#python train.py --backbone="resnet34" --model_path="trained_weight/celeba_unmd_resnet34/ep020-loss1.216-val_loss2.215.pth" --watermark_size=32 --annotation_path="celeba_cls_train.txt" --lfw_dir_path="datasets/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=30  --epoch=20




# CUDA_VISIBLE_DEVICES=3 python train.py --model_path="/home/zx/public/collaboration/facenet-pytorch/facenet-pytorch/trained_weight/faceweb_unmd_mobilenet/ep042-loss0.410-val_loss2.268.pth" --watermark_size=32 --annotation_path="cls_train.txt" --lfw_dir_path="/home/zx/public/dataset/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=9 --embed_128


################################loss function orignal facenet baseline
# CUDA_VISIBLE_DEVICES=3 python train.py --model_path="/home/lsf/facenet-pytorch/model_data/facenet_mobilenet.pth" --annotation_path="cls_train.txt" --lfw_dir_path="lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=690 --loss_baseline=True

# CUDA_VISIBLE_DEVICES=3 python train.py --model_path="model_data/facenet_mobilenet.pth" --annotation_path="cls_train.txt" --lfw_dir_path="/home/zx/public/dataset/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=690 --loss_baseline=True

# CUDA_VISIBLE_DEVICES=3 python train.py --model_path="model_data/facenet_mobilenet.pth" --annotation_path="cls_train.txt" --lfw_dir_path="/home/zx/public/dataset/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=90 --loss_baseline=True --loss_baseline_lambda=1.0 --epoch=10

# CUDA_VISIBLE_DEVICES=3 python train.py --model_path="model_data/facenet_mobilenet.pth" --annotation_path="cls_train.txt" --lfw_dir_path="/home/zx/public/dataset/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=90 --loss_baseline=True --loss_baseline_lambda=10.0 --epoch=10

# CUDA_VISIBLE_DEVICES=3 python train.py --model_path="model_data/facenet_mobilenet.pth" --annotation_path="cls_train.txt" --lfw_dir_path="/home/zx/public/dataset/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=90 --loss_baseline=True --loss_baseline_lambda=50.0 --epoch=10

# CUDA_VISIBLE_DEVICES=3 python train.py --model_path="model_data/facenet_mobilenet.pth" --annotation_path="cls_train.txt" --lfw_dir_path="/home/zx/public/dataset/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=90 --loss_baseline=True --loss_baseline_lambda=100.0 --epoch=10
# 


# CUDA_VISIBLE_DEVICES=0 python train.py --model_path="trained_weight/faceweb_unmd_mobilenet/ep042-loss0.410-val_loss2.268.pth" --annotation_path="faceweb_cls_train.txt" --lfw_dir_path="datasets/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=30 --loss_baseline=True --loss_baseline_lambda=50.0 --epoch=10

#CUDA_VISIBLE_DEVICES=0 python train.py --model_path="trained_weight/celeba_unmd_mobilenet/ep017-loss2.413-val_loss3.854.pth" --annotation_path="celeba_cls_train.txt" --lfw_dir_path="datasets/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=30 --loss_baseline=True --loss_baseline_lambda=50.0 --epoch=10

#PostNet
CUDA_VISIBLE_DEVICES=3 python train.py --model_path="" --watermark_size=32 --annotation_path="cls_train.txt" --lfw_dir_path="lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=96 --PostNet=True
