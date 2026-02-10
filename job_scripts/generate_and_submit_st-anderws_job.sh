#!/bin/bash

# 1. Check if an experiment name was provided (Required)
if [ -z "$1" ]; then
    echo "Error: No experiment name provided."
    echo "Usage: ./generate_and_submit_st-anderws_job.sh <exp_name> [cpus] [time_limit]"
    echo "Example: ./generate_and_submit_st-anderws_job.sh my_test 4 1-00:00:00"
    exit 1
fi

EXP_NAME=$1
# 2. Set defaults or use provided arguments
CPUS=${2:-8}            # Defaults to 8 if not provided
TIME_LIMIT=${3:-3-00:00:00} # Defaults to 3 days if not provided

OUT_DIR="./tmp"
SCRIPT_PATH="${OUT_DIR}/submit_${EXP_NAME}.sh"

mkdir -p "$OUT_DIR"

# 3. Create the file using a Heredoc
cat << EOF > "$SCRIPT_PATH"
#!/usr/bin/env bash
#SBATCH --job-name=${EXP_NAME}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=16G
#SBATCH --time=${TIME_LIMIT}
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-username@st-andrews.ac.uk
#SBATCH --output=${OUT_DIR}/%x-%j.log
#SBATCH --error=${OUT_DIR}/%x-%j.err
#SBATCH --partition=labruja
#SBATCH --gres=gpu:1

set -e

module purge
module load anaconda  

echo "Job ID: \$SLURM_JOB_ID"
echo "Allocated CPUs: ${CPUS}"
echo "Time Limit: ${TIME_LIMIT}"

cd \$HOME/project/bimanual_garment_folding
source ./setup.sh

echo "Starting training at \$(date)"

python ./tool/hydra_train.py \\
    --config-name sim_exp/${EXP_NAME}

echo "Job completed at \$(date)"
EOF

chmod +x "$SCRIPT_PATH"

echo "------------------------------------------------"
echo "Generated script: $SCRIPT_PATH"
echo "CPUs requested:   $CPUS"
echo "Time limit:       $TIME_LIMIT"
echo "------------------------------------------------"
echo "Submitting job..."
sbatch "$SCRIPT_PATH"