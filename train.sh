
source ~/anaconda3/etc/profile.d/conda.sh
cd /yqw/dev/SAS-VPReID

#Training was performed using ViT-Large-14.
CUDA_VISIBLE_DEVICES=0 python train_climb.py --config_file configs/vit_clipreid.yml
#Training was performed using ViT-Base-16.
CUDA_VISIBLE_DEVICES=0 python train_climb.py --config_file configs/vit_clipreid_base.yml