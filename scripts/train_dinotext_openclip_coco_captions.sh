#!/bin/bash
#SBATCH --job-name=dino_text_train
##SBATCH --job-name=debug
#SBATCH --output=/work/leonardo_phd/logs3/%x_%j.out
#SBATCH --error=/work/leonardo_phd/logs3/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=10G
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
##SBATCH --time=4:00:00
#SBATCH --partition=all_usr_prod
#SBATCH --account=pnrr_fit4medrob
##SBATCH --dependency=afterany:1772778
##SBATCH --exclude=aimagelab-srv-10,carabbaggio,vegeta,lurcanio,ajeje,germano,helmut
##SBATCH --array=0-1

export PYTHONNOUSERSITE=1
source activate dino-text

IFS=',' read -r -a nodelist <<<$SLURM_NODELIST
MASTER_ADDR="${nodelist[0]}"
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$(comm -23 <(seq 8000 9000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
# scontrol show job $SLURM_JOBID | grep Command >"$out_dir/command_script.txt"
# timestamp=$(date "+%Y%m%d-%H%M%S")
# echo $SLURM_JOBID >"$out_dir/slurm_jobid_${timestamp}.txt"

export MAX_RESTARTS=1
export WORLD_SIZE=$((SLURM_GPUS_PER_NODE * SLURM_NNODES))
# divide batch size by world size
batch_size=$((batch_size / WORLD_SIZE))

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "MAX_RESTARTS=$MAX_RESTARTS"

cd "/homes/lbarsellotti/projects/DINO-text/src/open_clip/src"

for i in $(seq 0 $((SLURM_NNODES - 1))); do
    srun --exclusive -N1 -n1 -w "${nodelist[$i]}" \
    python -u -m torch.distributed.run --nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_GPUS_PER_NODE --rdzv_id=$SLURM_JOB_ID --rdzv_backend="c10d" --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" --node_rank=$i --max_restarts=$MAX_RESTARTS \
    open_clip_train/main_dinotext.py \
    --save-frequency 1 \
    --train-data='/work/leonardo_phd/datasets/webdatasets/coco-384-training-{000..113}.tar' \
    --train-num-samples 566435 \
    --csv-img-key filepath \
    --csv-caption-key title \
    --warmup 0 \
    --batch-size=128 \
    --lr=1e-4 \
    --wd=0.1 \
    --epochs=20 \
    --workers=0 \
    --model RN50 \
    --model_cfg open_clip/model_configs/dinotext.yaml \
    --dataset-type=webdataset \
    --report-to 'wandb' \
    --wandb-project-name 'dino-text' \
    --local-loss \
    --gather-with-grad \
    --lr-scheduler const \
    --name 'const_lr_unfreeze_clip'
    sleep 5
done

wait # wait for all processes to finish