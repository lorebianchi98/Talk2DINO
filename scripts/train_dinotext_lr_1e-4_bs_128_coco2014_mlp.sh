#!/bin/bash
#SBATCH --job-name=train_dinotext_lr_1e-4_bs_128_coco2014_mlp
##SBATCH --job-name=debug
#SBATCH --output=/work/leonardo_phd/logs3/%x.out
#SBATCH --error=/work/leonardo_phd/logs3/%x.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=50G
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
##SBATCH --time=4:00:00
#SBATCH --partition=all_usr_prod
#SBATCH --account=pnrr_fit4medrob
##SBATCH --dependency=afterany:1772778
##SBATCH --exclude=aimagelab-srv-10,carabbaggio,vegeta,lurcanio,ajeje,germano,helmut
##SBATCH --array=0-2

export PYTHONNOUSERSITE=1
source activate dino-text

IFS=',' read -r -a nodelist <<<$SLURM_NODELIST
MASTER_ADDR="${nodelist[0]}"
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$(comm -23 <(seq 8000 9000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
# scontrol show job $SLURM_JOBID | grep Command >"$out_dir/command_script.txt"
# timestamp=$(date "+%Y%m%d-%H%M%S")
# echo $SLURM_JOBID >"$out_dir/slurm_jobid_${timestamp}.txt"

export MAX_RESTARTS=5
export WORLD_SIZE=$((SLURM_GPUS_PER_NODE * SLURM_NNODES))
# divide batch size by world size
batch_size=$((batch_size / WORLD_SIZE))

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "MAX_RESTARTS=$MAX_RESTARTS"

cd "/homes/lbarsellotti/projects/DINO-text"

python train.py --model_config configs/vitb_mlp_infonce_448_50-epochs_128-bs_lr-1e-4_coco2014.yaml --data_dir coco/ --train_dataset coco/coco_train_2014_original.pth --crop_dim 448 --test_dataset /home/lbarsellotti/projects/DINO-text/coco/test1k.json --val_dataset coco/coco_val_2014_original.pth --use_wandb