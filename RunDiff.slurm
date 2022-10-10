#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --job-name=Diffusion 
#SBATCH --time=24:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1

source /home/hpc/iwi5/iwi5044h/.bashrc
conda activate ldm
cd /home/hpc/iwi5/iwi5044h/latent-diffusion/
python main.py -b configs/latent-diffusion/nih-vq.yaml -t