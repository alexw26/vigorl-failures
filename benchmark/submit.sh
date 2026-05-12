#!/usr/bin/env bash
#SBATCH --job-name="vigorl_docvqa_benchmark"
#SBATCH --output="slurm_outputs/docvqa_%j.out"
#SBATCH --error="slurm_outputs/docvqa_%j.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest
#SBATCH --account=bheg-delta-gpu
#SBATCH -t 12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cui20@illinois.edu

PRJ_DIR="/u/sycui/dev/grounded-rl"

source "${PRJ_DIR}/.venv/bin/activate"
echo "$(type python)"
echo "" > "${OUTPUT_FILE}"

# python "${PRJ_DIR}/benchmark/benchmark_docvqa.py"
python "${PRJ_DIR}/benchmark/benchmark_haloquest.py"
