#!/usr/bin/env bash
#SBATCH --job-name="mirror_test"
#SBATCH --output="slurm_outputs/mirror_test_%j.out"
#SBATCH --error="slurm_outputs/mirror_test_%j.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest
#SBATCH --account=bheg-delta-gpu
#SBATCH -t 16:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cui20@illinois.edu


MODEL="gsarch/ViGoRL-7b-Spatial"
QUERY="Find a coordinate that is inside a mirror reflection."

PRJ_DIR="/u/sycui/dev/grounded-rl"
IMAGE_DIR="${PRJ_DIR}/benchmark/DLSU-OMRS/image"
OUTPUT_FILE="${PRJ_DIR}/benchmark/DLSU-OMRS/result.txt"

source "${PRJ_DIR}/.venv/bin/activate"
echo "$(type python)"
echo "" > "${OUTPUT_FILE}"

for i in {0..548}; do
    img="${IMAGE_DIR}/image-${i}.jpg"
    echo "Processing ${img}..."
    python "${PRJ_DIR}/demo/demo_singleturn.py" \
        --model "${MODEL}" \
        --image "${img}" \
        --query "${QUERY}" \
        >> "${OUTPUT_FILE}" 2>&1
done
