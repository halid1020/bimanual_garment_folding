#!/bin/bash
#SBATCH --job-name=folding_exp
#SBATCH --partition=gpu-l4-n1
#SBATCH --qos=gpu-l4-n1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/slurm-%x-%j.out

# Path to the image you will upload to the cluster
SIF_IMAGE="/home/$USER/st_andrews_cluster.sif"

# Note: We use absolute paths inside the container for the python script
apptainer exec --nv --writable-tmpfs $SIF_IMAGE \
    xvfb-run -a -s "-screen 0 640x480x24" \
    python3 /app/bimanual_garment_folding/train/hydra_eval.py \
    --config-name run_exp/random_multi_primitive_multi_longsleeve_folding_from_crumpled