#!/usr/bin/env bash

# --- Default Values ---
DAYS=3
TIME_STRING="" # We will calculate this based on DAYS or PARTITION
MEM="16G"
CPUS=4
PARTITION="gpu"
GRES="gpu:1"

# --- Help Function ---
usage() {
    echo "Usage: $0 <experiment_name> [options]"
    echo "Options:"
    echo "  -t  Days (Default: 3)"
    echo "  -m  Memory (Default: 16G)"
    echo "  -c  CPUs per task (Default: 4)"
    echo "  -p  Partition (Default: gpu)"
    echo "  -g  GRES/GPU config (Default: gpu:1)"
    exit 1
}

if [ -z "$1" ] || [[ "$1" == -* ]]; then
    echo "Error: Experiment name must be the first argument."
    usage
fi

EXP_NAME=$1
shift

while getopts "t:m:c:p:g:" opt; do
  case $opt in
    t) DAYS=$OPTARG ;;
    m) MEM=$OPTARG ;;
    c) CPUS=$OPTARG ;;
    p) PARTITION=$OPTARG ;;
    g) GRES=$OPTARG ;;
    *) usage ;;
  esac
done

# --- 🚀 AUTOMATIC TIME ADJUSTMENT ---
if [ "$PARTITION" == "gpu_short" ]; then
    echo "💡 gpu_short partition detected: Setting time limit to 20 minutes."
    TIME_STRING="00:20:00"
else
    # Standard format: Days-HH:MM:SS
    TIME_STRING="${DAYS}-00:00:00"
fi

OUT_DIR="./tmp"
SCRIPT_PATH="${OUT_DIR}/submit_${EXP_NAME}.sh"
mkdir -p "$OUT_DIR"

cat << EOF > "$SCRIPT_PATH"
#!/usr/bin/env bash
#SBATCH --job-name=${EXP_NAME}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME_STRING}
#SBATCH --account=cs-garm-2025
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hcv530@york.ac.uk
#SBATCH --output=${OUT_DIR}/%x-%j.log
#SBATCH --error=${OUT_DIR}/%x-%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}

set -e

module purge
module load Miniconda3/23.5.2-0
source ~/.bashrc

# Force use of the NVIDIA vendor library for OpenGL
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __NV_PRIME_RENDER_OFFLOAD=1
export __VK_LAYER_NV_optimus=NVIDIA_only

# Tell the driver not to look for a physical screen
export QT_QPA_PLATFORM=offscreen
export SDL_VIDEODRIVER=dummy


echo "Job ID: \$SLURM_JOB_ID"
nvidia-smi

cd /users/hcv530/project/bimanual_garment_folding
# Note: Using your new renamed env setup logic
source ./setup.sh

echo "Starting evaluation at \$(date)"
python ./tool/hydra_train.py --config-name run_exp/${EXP_NAME}
echo "Job completed at \$(date)"
EOF

chmod +x "$SCRIPT_PATH"

echo "------------------------------------------------"
echo "Experiment: $EXP_NAME"
echo "Config:     Time=${TIME_STRING}, Mem=${MEM}, Partition=${PARTITION}"
echo "Submitting..."
sbatch "$SCRIPT_PATH"