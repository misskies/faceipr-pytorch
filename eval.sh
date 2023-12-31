#CUDA_VISIBLE_DEVICES=0 python eval_LFW.py --model_path="/home/lsf/facenet-pytorch/trained_weight/faceweb_md_mobilenet_Decoder-5FC/32/ep013-loss0.732-val_loss2.137.pth" --watermark_size=32 --robustness="combine"
#CUDA_VISIBLE_DEVICES=3 python eval_LFW.py --batch_size=256 --model_path="/home/lsf/facenet-pytorch/trained_weight/faceweb_md_mobilenet_Decoder-5FC/32/ep013-loss0.732-val_loss2.137.pth" --watermark_size=32 --robustness="round" --test_rank=12 --noise_power=11
#CUDA_VISIBLE_DEVICES=3 python eval_LFW.py --batch_size=256 --model_path="/home/lsf/facenet-pytorch/trained_weight/faceweb_md_mobilenet_Decoder-5FC/32/ep013-loss0.732-val_loss2.137.pth" --watermark_size=32 --robustness="random_del" --test_rank=21

#LSB
#CUDA_VISIBLE_DEVICES=0 python eval_LFW.py --model_path="/home/lsf/facenet-pytorch/trained_weight/faceweb_unmd_mobilenet/ep042-loss0.410-val_loss2.268.pth" --LSB=True --robustness="none"
#CUDA_VISIBLE_DEVICES=0 python eval_LFW.py --model_path="/home/lsf/facenet-pytorch/trained_weight/faceweb_unmd_mobilenet/ep042-loss0.410-val_loss2.268.pth" --LSB=True --robustness="noise"
#CUDA_VISIBLE_DEVICES=0 python eval_LFW.py --model_path="/home/lsf/facenet-pytorch/trained_weight/faceweb_unmd_mobilenet/ep042-loss0.410-val_loss2.268.pth" --LSB=True --robustness="combine"
#CUDA_VISIBLE_DEVICES=3 python eval_LFW.py --batch_size=256 --model_path="/home/lsf/facenet-pytorch/trained_weight/faceweb_unmd_mobilenet/ep042-loss0.410-val_loss2.268.pth" --watermark_size=32 --robustness="round" --test_rank=12 --noise_power=11 --LSB=True
#CUDA_VISIBLE_DEVICES=3 python eval_LFW.py --batch_size=256 --model_path="/home/lsf/facenet-pytorch/trained_weight/faceweb_unmd_mobilenet/ep042-loss0.410-val_loss2.268.pth" --watermark_size=32 --robustness="noise" --test_rank=21 --LSB=True
#CUDA_VISIBLE_DEVICES=3 python eval_LFW.py --batch_size=256 --model_path="/home/lsf/facenet-pytorch/trained_weight/celeba_unmd_mobilenet/ep017-loss2.413-val_loss3.854.pth" --watermark_size=32 --robustness="none" --test_rank=1 --LSB=True



#loss function baseline
#CUDA_VISIBLE_DEVICES=3 python eval_LFW.py --batch_size=256 --model_path="/home/lsf/public/collaboration/facenet-pytorch/facenet-pytorch/trained_weight/faceweb_lossEmbed_mobilenet/ep010-loss42.281-val_loss43.584.pth" --robustness="noise" --test_rank=21 --loss_baseline=True
#CUDA_VISIBLE_DEVICES=3 python eval_LFW.py --batch_size=256 --model_path="/home/lsf/public/collaboration/facenet-pytorch/facenet-pytorch/trained_weight/faceweb_lossEmbed_mobilenet/ep010-loss42.281-val_loss43.584.pth" --robustness="round" --test_rank=12 --noise_power=11 --loss_baseline=True
#CUDA_VISIBLE_DEVICES=3 python eval_LFW.py --batch_size=128 --model_path="/home/lsf/public/collaboration/facenet-pytorch/facenet-pytorch/trained_weight/faceweb_lossEmbed_mobilenet/ep010-loss42.281-val_loss43.584.pth" --robustness="random_del" --test_rank=21 --loss_baseline=True

#Confrontation training eval
#CUDA_VISIBLE_DEVICES=3 python eval_LFW.py --batch_size=256 --model_path="/home/lsf/facenet-pytorch/logs/noise-0.2/ep005-loss0.676-val_loss2.262.pth" --watermark_size=32 --robustness="noise"  --test_rank=21

#CUDA_VISIBLE_DEVICES=3 python eval_LFW.py --batch_size=128 --model_path="/home/lsf/facenet-pytorch/trained_weight/faceweb_md_mobilenet_Decoder-5FC/32/ep013-loss0.732-val_loss2.137.pth" --watermark_size=32 --robustness="random_del" --test_rank=21
#CUDA_VISIBLE_DEVICES=3 python eval_LFW.py --batch_size=128 --model_path="/home/lsf/facenet-pytorch/trained_weight/faceweb_md_mobilenet_Decoder-5FC/32/ep013-loss0.732-val_loss2.137.pth" --watermark_size=32 --robustness="combine" --test_rank=21
#
#CUDA_VISIBLE_DEVICES=3 python eval_LFW.py --batch_size=128 --model_path="/home/lsf/public/collaboration/facenet-pytorch/facenet-pytorch/trained_weight/faceweb_lossEmbed_mobilenet/ep010-loss42.281-val_loss43.584.pth" --robustness="random_del" --test_rank=21 --loss_baseline=True
#CUDA_VISIBLE_DEVICES=3 python eval_LFW.py --batch_size=128 --model_path="/home/lsf/public/collaboration/facenet-pytorch/facenet-pytorch/trained_weight/faceweb_lossEmbed_mobilenet/ep010-loss42.281-val_loss43.584.pth" --robustness="combine" --test_rank=21 --loss_baseline=True
#
#CUDA_VISIBLE_DEVICES=0 python eval_LFW.py --batch_size=128 --model_path="/home/lsf/facenet-pytorch/trained_weight/faceweb_unmd_mobilenet/ep042-loss0.410-val_loss2.268.pth" --robustness="random_del" --test_rank=21 --post=LSB --watermark_size=1024
#CUDA_VISIBLE_DEVICES=0 python eval_LFW.py --batch_size=128 --model_path="/home/lsf/facenet-pytorch/trained_weight/faceweb_unmd_mobilenet/ep042-loss0.410-val_loss2.268.pth" --robustness="combine" --test_rank=21 --post=LSB --watermark_size=1024
#
#CUDA_VISIBLE_DEVICES=3 python eval_LFW.py --batch_size=128 --model_path="/home/lsf/facenet-pytorch/trained_weight/faceweb_unmd_mobilenet/ep042-loss0.410-val_loss2.268.pth" --robustness="random_del" --test_rank=21 --post=FFT
#CUDA_VISIBLE_DEVICES=3 python eval_LFW.py --batch_size=128 --model_path="/home/lsf/facenet-pytorch/trained_weight/faceweb_unmd_mobilenet/ep042-loss0.410-val_loss2.268.pth" --robustness="combine" --test_rank=21 --post=FFT

# adv eval
#CUDA_VISIBLE_DEVICES=3 python eval_LFW.py --batch_size=256 --model_path="/home/lsf/facenet-pytorch/logs/random_del-0.1/ep028-loss0.348-val_loss2.056.pth" --watermark_size=32 --robustness="random_del"  --test_robustness="ADV_random_del-0.1"
#CUDA_VISIBLE_DEVICES=3 python eval_LFW.py --batch_size=256 --model_path="/home/lsf/facenet-pytorch/logs/random_del-0.05/ep006-loss0.624-val_loss2.171.pth" --watermark_size=32 --robustness="random_del" --test_robustness="ADV_random_del-0.05"
#
#CUDA_VISIBLE_DEVICES=3 python eval_LFW.py --batch_size=256 --model_path="/home/lsf/facenet-pytorch/logs/combine-0.1/ep001-loss1.400-val_loss2.374.pth" --watermark_size=32 --robustness="combine"  --test_robustness="ADV_combine-0.1"
#CUDA_VISIBLE_DEVICES=3 python eval_LFW.py --batch_size=256 --model_path="/home/lsf/facenet-pytorch/logs/combine-0.05/ep002-loss0.650-val_loss1.935.pth" --watermark_size=32 --robustness="combine"  --test_robustness="ADV_combine-0.05"

#PostNet
CUDA_VISIBLE_DEVICES=0 python eval_LFW.py --batch_size=128 --model_path="/home/lsf/facenet-pytorch/logs/loss_2023_09_02_20_13_38/ep011-loss0.012-val_loss0.120.pth" --watermark_size=32  --test_rank=1 --PostNet=True
