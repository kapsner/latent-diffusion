#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --job-name=DiffusionAE 
#SBATCH --time=24:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH -C a100_80

source /home/hpc/iwi5/iwi5044h/.bashrc
conda activate ldm
cd /home/hpc/iwi5/iwi5044h/latent-diffusion/
python main.py -b configs/autoencoder/ae_nih.yaml -t -r /home/hpc/iwi5/iwi5044h/latent-diffusion/logs/2022-10-08T14-07-28_ae_nih/checkpoints/last.ckpt
