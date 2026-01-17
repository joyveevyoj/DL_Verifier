#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=8g
#SBATCH --gpus=4
#SBATCH --gres=gpumem:24g
#SBATCH -A es_hutter
#SBATCH --time=48:00:00
#SBATCH --tmp=100g
#SBATCH --mail-type=NONE
#SBATCH --mail-user=fetian@ethz.ch
#SBATCH --job-name="train-both"
#SBATCH --output=logs/%x_%j.out

# Load modules
module load eth_proxy

# Job information
echo "========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Starting at: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /cluster/project/rsl/$USER/miniconda3/bin/activate
conda activate llm-verifier

cd /cluster/home/$USER/llm-verifier

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "Master Port: $MASTER_PORT"

accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --mixed_precision bf16 \
    --main_process_port $MASTER_PORT \
    scripts/main.py \
    --mode both

echo "Job completed at $(date)"
