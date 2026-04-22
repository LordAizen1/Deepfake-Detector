#!/bin/bash
#SBATCH --job-name=deepfake-train
#SBATCH --output=%j_out.log
#SBATCH --error=%j_error.log
#SBATCH --time=06:00:00
#SBATCH --qos=medium
#SBATCH --partition=medium
#SBATCH --mem=32G
#SBATCH --account=ravi
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=gpu02
#SBATCH --gres=gpu:H100:1

source ~/miniconda3/bin/activate deepfake
export LD_LIBRARY_PATH=$(python -c "import nvidia.cudnn; import os; print(os.path.dirname(nvidia.cudnn.__file__) + '/lib')"):$LD_LIBRARY_PATH
export WANDB_MODE=offline
cd ~/Deepfake-Detector/backend
python train.py
