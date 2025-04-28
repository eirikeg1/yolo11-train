#!/bin/bash
#SBATCH --job-name=train_video_gat
#SBATCH --account=ec395
#SBATCH --mail-type=FAIL
#SBATCH --partition=accel_long
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=12
#SBATCH --time=47:59:00
#SBATCH --gres=gpu:a100:1
#========================================
# #SBATCH --gpus=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --qos=devel
# #SBATCH --mem-per-gpu=40G
# #SBATCH --gpus-per-node=1
# # SBATCH --partition=accel
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

nvidia-smi

module --quiet purge
module --latest load Miniconda3
# module load CUDA/12.1.1

source /cluster/software/EL9/easybuild/software/Miniconda3/22.11.1-1/etc/profile.d/conda.sh

# Initialize Conda for non-interactive shells
eval "$(conda shell.bash hook)"

echo "conda envs"
conda info --envs

conda activate /fp/homes01/u01/ec-eirikeg/.conda/envs/dribling-detection

python train_yolo.py --out_path "${SCRATCH}" $@






