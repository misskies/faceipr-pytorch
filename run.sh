# python -m torch.distributed.launch train.py --model_path="" --watermark_size=32 --annotation_path="cls_train.txt" --lfw_dir_path="/home/zx/public/dataset/lfw" --lfw_pairs_path="model_data/lfw_pair.txt"
# python -m torch.distributed.launch train.py --model_path="" --watermark_size=32 --annotation_path="cls_train.txt" --lfw_dir_path="/home/zx/public/dataset/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=9
# python -m torch.distributed.launch train.py --model_path="" --watermark_size=32 --annotation_path="cls_train.txt" --lfw_dir_path="/home/zx/public/dataset/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=9 --robustness="flip"

#CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=1 train.py --model_path="" --watermark_size=32 --annotation_path="cls_train.txt" --lfw_dir_path="/home/zx/public/dataset/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=9 --robustness="combine"

################################1. no modulation on watermark_mobilenet
# python train.py --model_path="" --annotation_path="cls_train.txt" --lfw_dir_path="/home/zx/public/dataset/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=9 --original=True

################################2.load no modulation model + encoder + decoder
# python train.py --model_path="/home/lsf/facenet-pytorch/trained_weight/faceweb_unmd_mobilenet/ep042-loss0.410-val_loss2.268.pth" --watermark_size==32 --annotation_path="cls_train.txt" --lfw_dir_path="/home/zx/public/dataset/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=9 --epoch=10

# python train.py --model_path="/home/lsf/facenet-pytorch/trained_weight/faceweb_unmd_mobilenet/ep042-loss0.410-val_loss2.268.pth" --watermark_size==64 --annotation_path="cls_train.txt" --lfw_dir_path="/home/zx/public/dataset/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=9  --epoch=50

# CUDA_VISIBLE_DEVICES=3 python train.py --model_path="/home/zx/public/collaboration/facenet-pytorch/facenet-pytorch/trained_weight/faceweb_unmd_mobilenet/ep042-loss0.410-val_loss2.268.pth" --watermark_size=32 --annotation_path="cls_train.txt" --lfw_dir_path="/home/zx/public/dataset/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=9 --embed_128


################################loss function orignal facenet baseline
# CUDA_VISIBLE_DEVICES=3 python train.py --model_path="/home/lsf/facenet-pytorch/model_data/facenet_mobilenet.pth" --annotation_path="cls_train.txt" --lfw_dir_path="lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=690 --loss_baseline=True

# CUDA_VISIBLE_DEVICES=3 python train.py --model_path="model_data/facenet_mobilenet.pth" --annotation_path="cls_train.txt" --lfw_dir_path="/home/zx/public/dataset/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=690 --loss_baseline=True

# CUDA_VISIBLE_DEVICES=3 python train.py --model_path="model_data/facenet_mobilenet.pth" --annotation_path="cls_train.txt" --lfw_dir_path="/home/zx/public/dataset/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=90 --loss_baseline=True --loss_baseline_lambda=1.0 --epoch=10

# CUDA_VISIBLE_DEVICES=3 python train.py --model_path="model_data/facenet_mobilenet.pth" --annotation_path="cls_train.txt" --lfw_dir_path="/home/zx/public/dataset/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=90 --loss_baseline=True --loss_baseline_lambda=10.0 --epoch=10

# CUDA_VISIBLE_DEVICES=3 python train.py --model_path="model_data/facenet_mobilenet.pth" --annotation_path="cls_train.txt" --lfw_dir_path="/home/zx/public/dataset/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=90 --loss_baseline=True --loss_baseline_lambda=50.0 --epoch=10

# CUDA_VISIBLE_DEVICES=3 python train.py --model_path="model_data/facenet_mobilenet.pth" --annotation_path="cls_train.txt" --lfw_dir_path="/home/zx/public/dataset/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=90 --loss_baseline=True --loss_baseline_lambda=100.0 --epoch=10
# 


CUDA_VISIBLE_DEVICES=3 python train.py --model_path="/home/zx/public/collaboration/facenet-pytorch/facenet-pytorch/trained_weight/faceweb_unmd_mobilenet/ep042-loss0.410-val_loss2.268.pth" --annotation_path="cls_train.txt" --lfw_dir_path="/home/zx/public/dataset/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=30 --loss_baseline=True --loss_baseline_lambda=50.0 --epoch=10

CUDA_VISIBLE_DEVICES=3 python train.py --model_path="/home/zx/public/collaboration/facenet-pytorch/facenet-pytorch/trained_weight/faceweb_unmd_mobilenet/ep042-loss0.410-val_loss2.268.pth" --annotation_path="cls_train_celeba.txt" --lfw_dir_path="/home/zx/public/dataset/lfw" --lfw_pairs_path="model_data/lfw_pair.txt" --batch_size=30 --loss_baseline=True --loss_baseline_lambda=50.0 --epoch=10
