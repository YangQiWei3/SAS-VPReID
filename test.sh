
source ~/anaconda3/etc/profile.d/conda.sh
cd /yqw/dev/SAS-VPReID

CUDA_VISIBLE_DEVICES=0 python evaluate_all_cases.py --config_file configs/vit_clipreid.yml --model_path output_original/ViT-L-14-no_26.pth
CUDA_VISIBLE_DEVICES=0 python evaluate_all_cases.py --config_file configs/vit_clipreid.yml --model_path output_original/ViT-L-14-no_27.pth
CUDA_VISIBLE_DEVICES=0 python evaluate_all_cases.py --config_file configs/vit_clipreid.yml --model_path output_original/ViT-L-14-no_28.pth
CUDA_VISIBLE_DEVICES=0 python evaluate_all_cases.py --config_file configs/vit_clipreid.yml --model_path output_original/ViT-L-14-no_29.pth
