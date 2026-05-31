#!/bin/bash
#SBATCH --job-name=ella
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

set -e

cd /scratch/workspace/hongxinzhang_umass_edu-shared/Ella
source .venv/bin/activate

mkdir -p logs

SCRIPT=${1:-scripts/IB/test_IB_ella_seg_newyork.sh}
SCRIPT_NAME=$(basename "$SCRIPT" .sh)
scontrol update JobId="$SLURM_JOB_ID" JobName="ella-${SCRIPT_NAME}"
ln -sf "slurm_${SLURM_JOB_ID}.out" "logs/${SCRIPT_NAME}_${SLURM_JOB_ID}.out"
ln -sf "slurm_${SLURM_JOB_ID}.err" "logs/${SCRIPT_NAME}_${SLURM_JOB_ID}.err"
bash "$SCRIPT"
