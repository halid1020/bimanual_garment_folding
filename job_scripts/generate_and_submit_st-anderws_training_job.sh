#!/bin/bash

# --- Defaults ---
JOB_TYPE="train"
MEMORY="16G"
TIME_LIMIT="3-00:00:00"
CPUS=8 

# 1. Parse flags
while getopts "j:m:t:c:" opt; do
  case $opt in
    j) JOB_TYPE=$OPTARG ;;
    m) MEMORY=$OPTARG ;;
    t) TIME_LIMIT=$OPTARG ;;
    c) CPUS=$OPTARG ;; # Optional: Added flag for CPUs just in case
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
  esac
done

# Shift so $1 becomes the experiment name
shift $((OPTIND-1))

# 2. Check for Experiment Name
if [ -z "$1" ]; then
    echo "Error: No experiment name provided."
    echo "Usage: $0 [-j job_type] [-m memory] [-t time] [-c cpus] <exp_name>"
    echo "Example: $0 -j collect -m 32G -t 1-12:00:00 -c 16 my_experiment"
    exit 1
fi

EXP_NAME=$1

# 3. specific logic based on Job Type
case $JOB_TYPE in
    train)
        SCRIPT_NAME="hydra_train.py"
        CONFIG_DIR="sim_exp"
        ;;
    eval)
        SCRIPT_NAME="hydra_eval.py"
        CONFIG_DIR="sim_exp"
        ;;
    collect)
        SCRIPT_NAME="hydra_collect.py"
        CONFIG_DIR="data_collection"
        ;;
    *)
        echo "Error: Invalid job type '$JOB_TYPE'. Allowed: train, eval, collect."
        exit 1
        ;;
esac

# Define paths
OUT_DIR="/data/ah390/exp_logs"
SUBMIT_SCRIPT="${OUT_DIR}/submit_${JOB_TYPE}_${EXP_NAME}.sh"
mkdir -p "$OUT_DIR"

# 4. Generate the SLURM script
cat << EOF > "$SUBMIT_SCRIPT"
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_TYPE}_${EXP_NAME}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-username@st-andrews.ac.uk
#SBATCH --output=${OUT_DIR}/%x-%j.log
#SBATCH --error=${OUT_DIR}/%x-%j.err
#SBATCH --partition=labruja
#SBATCH --gres=gpu:1 

echo "Job ID: \$SLURM_JOB_ID"
echo "Job Type: ${JOB_TYPE}"
echo "CPUs: ${CPUS}, Mem: ${MEMORY}, Time: ${TIME_LIMIT}"

cd \$HOME/project/bimanual_garment_folding
source ./setup.sh

echo "Starting ${JOB_TYPE}..."

# Using the dynamic script name and config directory
python ./tool/${SCRIPT_NAME} config_name=${CONFIG_DIR}/${EXP_NAME}

echo "Job completed at \$(date)"
EOF

chmod +x "$SUBMIT_SCRIPT"

echo "------------------------------------------------"
echo "Generated:  $SUBMIT_SCRIPT"
echo "Job Type:   $JOB_TYPE"
echo "Config Dir: conf/$CONFIG_DIR"
echo "Script:     tool/$SCRIPT_NAME"
echo "Resources:  $CPUS CPUs, $MEMORY RAM, $TIME_LIMIT"
echo "------------------------------------------------"
echo "Submitting job..."
sbatch "$SUBMIT_SCRIPT"