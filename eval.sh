#CUDA_VISIBLE_DEVICES=0 python eval_LFW.py --model_path="/home/lsf/facenet-pytorch/trained_weight/faceweb_md_mobilenet_Decoder-5FC/32/ep013-loss0.732-val_loss2.137.pth" --watermark_size=32 --robustness="combine"

#LSB
#CUDA_VISIBLE_DEVICES=0 python eval_LFW.py --model_path="/home/lsf/facenet-pytorch/trained_weight/faceweb_unmd_mobilenet/ep042-loss0.410-val_loss2.268.pth" --LSB=True --robustness="none"
#CUDA_VISIBLE_DEVICES=0 python eval_LFW.py --model_path="/home/lsf/facenet-pytorch/trained_weight/faceweb_unmd_mobilenet/ep042-loss0.410-val_loss2.268.pth" --LSB=True --robustness="noise"
# CUDA_VISIBLE_DEVICES=0 python eval_LFW.py --model_path="/home/lsf/facenet-pytorch/trained_weight/faceweb_unmd_mobilenet/ep042-loss0.410-val_loss2.268.pth" --LSB=True --robustness="combine"



#FFT
CUDA_VISIBLE_DEVICES=0 python eval_LFW.py --model_path="/home/zx/public/collaboration/facenet-pytorch/facenet-pytorch/trained_weight/faceweb_unmd_mobilenet/ep042-loss0.410-val_loss2.268.pth" --lfw_dir_path="/home/zx/public/dataset/lfw"   --post="Noise" --robustness="none" --batch_size=9 --watermark_size=1024


