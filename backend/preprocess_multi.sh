#!/bin/bash
#SBATCH --job-name=deepfake-preprocess-multi
#SBATCH --output=%A_%a_out.log
#SBATCH --error=%A_%a_error.log
#SBATCH --time=01:30:00
#SBATCH --qos=medium
#SBATCH --partition=medium
#SBATCH --mem=32G
#SBATCH --account=ravi
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=gpu04
#SBATCH --gres=gpu:1
#SBATCH --array=0-3

MANIP_TYPES=("face2face" "faceswap" "faceshifter" "neuraltextures")
MANIP=${MANIP_TYPES[$SLURM_ARRAY_TASK_ID]}

source ~/miniconda3/bin/activate deepfake
export LD_LIBRARY_PATH=$(python -c "import nvidia.cudnn; import os; print(os.path.dirname(nvidia.cudnn.__file__) + '/lib')"):$LD_LIBRARY_PATH
cd ~/Deepfake-Detector/backend

echo "Processing: $MANIP"
python preprocess_multi.py $MANIP
