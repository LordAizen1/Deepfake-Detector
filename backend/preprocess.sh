#!/bin/bash
#SBATCH --job-name=deepfake-preprocess
#SBATCH --output=%j_out.log
#SBATCH --error=%j_error.log
#SBATCH --time=02:00:00
#SBATCH --qos=medium
#SBATCH --partition=medium
#SBATCH --mem=16G
#SBATCH --account=ravi
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=gpu04
#SBATCH --gres=gpu:1

source ~/miniconda3/bin/activate deepfake
cd ~/Deepfake-Detector/backend
python preprocess.py
