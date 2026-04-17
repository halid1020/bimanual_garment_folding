#!/usr/bin/env bash

# --- Default Values ---
DAYS=3
TIME_STRING="" # Calculated later based on input or partition
MEM="16G"
CPUS=4
PARTITION="gpu"
GRES="gpu:1"
PY_SCRIPT="hydra_train.py" # Default script to run

# --- Help Function ---
usage() {
    echo "Usage: $0 <experiment_name> [options]"
    echo "Options:"
    echo "  -t  Time limit (e.g., '3' for 3 days, '30m' for 30 mins, or '02:00:00') (Default: 3 days)"
    echo "  -m  Memory (Default: 16G)"
    echo "  -c  CPUs per task (Default: 4)"
    echo "  -p  Partition (Default: gpu)"
    echo "  -g  GRES/GPU config (Default: gpu:1)"
    echo "  -e  Run hydra_eval.py instead of hydra_train.py"
    exit 1
}

if [ -z "$1" ] || [[ "$1" == -* ]]; then
    echo "Error: Experiment name must be the first argument."
    usage
fi

EXP_NAME=$1
shift

while getopts "t:m:c:p:g:e" opt; do
  case $opt in
    t) 
      # Smart time parsing
      if [[ "$OPTARG" == *m ]]; then
          MINS=${OPTARG%m}
          TIME_STRING="00:${MINS}:00"
      elif [[ "$OPTARG" == *:* ]] || [[ "$OPTARG" == *-* ]]; then
          TIME_STRING="$OPTARG"
      else
          DAYS="$OPTARG"
      fi
      ;;
    m) MEM=$OPTARG ;;
    c) CPUS=$OPTARG ;;
    p) PARTITION=$OPTARG ;;
    g) GRES=$OPTARG ;;
    e) PY_SCRIPT="hydra_eval.py" ;; # Switch to eval script
    *) usage ;;
  esac
done

# --- 🚀 AUTOMATIC TIME ADJUSTMENT ---
if [ "$PARTITION" == "gpu_short" ]; then
    echo "💡 gpu_short partition detected: Setting time limit to 20 minutes."
    TIME_STRING="00:20:00"
elif [ -z "$TIME_STRING" ]; then
    # Fallback to Days if no specific time string was built from the -t flag
    TIME_STRING="${DAYS}-00:00:00"
fi

OUT_DIR="./tmp"
SAFE_EXP_NAME="${EXP_NAME//\//_}"
SCRIPT_PATH="${OUT_DIR}/submit_${SAFE_EXP_NAME}.sh"

mkdir -p "$OUT_DIR"

cat << EOF > "$SCRIPT_PATH"
#!/usr/bin/env bash
#SBATCH --job-name=${SAFE_EXP_NAME}
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

echo "Starting execution at \$(date)"
python ./tool/${PY_SCRIPT} --config-name sim_exp/${EXP_NAME}
echo "Job completed at \$(date)"
EOF

chmod +x "$SCRIPT_PATH"

echo "------------------------------------------------"
echo "Experiment: $EXP_NAME"
echo "Target script: $PY_SCRIPT"
echo "Config:     Time=${TIME_STRING}, Mem=${MEM}, Partition=${PARTITION}"
echo "Submitting..."
sbatch "$SCRIPT_PATH"