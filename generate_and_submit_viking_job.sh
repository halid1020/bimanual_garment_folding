#!/usr/bin/env bash

# 1. Check if an experiment name was provided
if [ -z "$1" ]; then
    echo "Error: No experiment name provided."
    echo "Usage: ./generate_and_submit_viking_job.sh <experiment_name> [days]"
    exit 1
fi

EXP_NAME=$1
# 2. Get days from the second argument, defaulting to 3 if not provided
DAYS=${2:-3}

OUT_DIR="./tmp"
SCRIPT_PATH="${OUT_DIR}/submit_${EXP_NAME}.sh"

# 3. Create the output directory if it doesn't exist
mkdir -p "$OUT_DIR"

# 4. Create the file using a Heredoc
cat << EOF > "$SCRIPT_PATH"
#!/usr/bin/env bash
#SBATCH --job-name=${EXP_NAME}
#SBATCH --nodes=1                            # Number of nodes to run on
#SBATCH --ntasks=1                           # Number of MPI tasks to request
#SBATCH --cpus-per-task=4                    # Number of CPU cores per MPI task
#SBATCH --mem=16G                            # Total memory to request
#SBATCH --time=${DAYS}-00:00:00              # Time limit (DD-HH:MM:SS)
#SBATCH --account=cs-garm-2025               # Project account to use
#SBATCH --mail-type=END,FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=hcv530@york.ac.uk        # Where to send mail
#SBATCH --output=${OUT_DIR}/%x-%j.log        # Standard output log (saved to ./tmp)
#SBATCH --error=${OUT_DIR}/%x-%j.err         # Standard error log (saved to ./tmp)
#SBATCH --partition=gpuplus
#SBATCH --gres=gpu:1

# Abort if any command fails
set -e

# 1. Clean Environment
module purge

# 2. Load Conda
module load Miniconda3/23.5.2-0

# 3. Activate your specific environment
source ~/.bashrc

# 4. Diagnostics
echo "Job ID: \$SLURM_JOB_ID"
echo "Running on host: \$(hostname)"
echo "Working directory: \$(pwd)"
nvidia-smi  # Check if GPU is visible

# 5. Project Setup
cd /users/hcv530/project/bimanual_garment_folding
source ./setup.sh

# 6. Run Training
echo "Starting training at \$(date)"

python ./tool/hydra_train.py \\
    --config-name run_exp/${EXP_NAME}

echo "Job completed at \$(date)"
EOF

# 5. Make the generated file executable
chmod +x "$SCRIPT_PATH"

# 6. Submit the job immediately
echo "Generated script at: $SCRIPT_PATH (Requested Time: ${DAYS} days)"
echo "Submitting job..."
sbatch "$SCRIPT_PATH"