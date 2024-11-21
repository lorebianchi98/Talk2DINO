#!/bin/bash
# --- PROD ---
#SBATCH --job-name=avg_self_attn_features
#SBATCH --output=/work/leonardo_phd/logs2/%x_%j_%a.out
#SBATCH --error=/work/leonardo_phd/logs2/%x_%j_%a.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=6G
##SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --partition=all_usr_prod
#SBATCH --account=pnrr_fit4medrob
#SBATCH --dependency=afterany:2333738
##SBATCH --exclude=huber,lurcanio,germano,carabbaggio,vegeta,ajeje,helmut,aimagelab-srv-10
##SBATCH --exclude=lurcanio,germano,carabbaggio,vegeta,ajeje,helmut,aimagelab-srv-10
##SBATCH --exclude=lurcanio
##SBATCH --nodelist=ailb-login-03,ajeje,carabbaggio,germano,gervasoni,helmut,huber,nico,nullazzo,rezzonico,tafazzi,pippobaudo,vegeta
##SBATCH --qos=all_qos_lowprio_short
##SBATCH --nodelist=tafazzi
##SBATCH --array=0-1192%25
#SBATCH --constraint="gpu_1080_8G|gpu_RTX5000_16G|gpu_RTX6000_24G|gpu_RTXA5000_24G|gpu_A40_48G"

source activate emiglio

proj_dir="/homes/lbarsellotti/projects/DINO-text"
cd $proj_dir

export PYTHONPATH="$proj_dir:$PYTHONPATH"
export PYTHONNOUSERSITE=1

srun --exclusive \
    python -u dino_extraction.py \
    --data_dir coco/ --batch_size 32 --extract_avg_self_attn --ann_path coco/train.json
