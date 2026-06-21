#!/bin/bash

# 1. Check if exactly one argument (the experiment name) is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <experiment_name>"
    echo "Example: $0 magpie_ctr_align_all_sim_garments_p4_v105"
    exit 1
fi

EXP_NAME=$1

# 2. Define the target directories
SIM_EXP_DIR="conf/sim_exp/magpie"
TRANSFER_EVAL_DIR="conf/transfer_eval/magpie"

# 3. Create the directories if they do not exist (-p ensures no error if they do)
mkdir -p "$SIM_EXP_DIR"
mkdir -p "$TRANSFER_EVAL_DIR"

# 4. Define the file paths
SIM_EXP_FILE="$SIM_EXP_DIR/$EXP_NAME.yaml"
TRANSFER_EVAL_FILE="$TRANSFER_EVAL_DIR/$EXP_NAME.yaml"

# 5. Generate the sim_exp YAML file
cat <<EOF > "$SIM_EXP_FILE"
# @package _global_
defaults:
  - /agent/magpie@agent: $EXP_NAME
  - /arena/magpie@arena: multi_longsleeve_provide_semkey_pixel_no_success_stop_resol_128_workspace
  - /task/magpie@task: central_alignment

exp_name: $EXP_NAME
project_name: bimanual_garment_folding
save_root: /mnt/ssd/garment_folding_data
train_and_eval: train_and_evaluate_single
EOF

echo "✅ Created: $SIM_EXP_FILE"

# 6. Generate the transfer_eval YAML file
cat <<EOF > "$TRANSFER_EVAL_FILE"
# @package transfer_eval

# Pass the training config as a string so Hydra doesn't auto-merge it and break the namespace
train_exp_config: sim_exp/magpie/$EXP_NAME

# Define evaluation arenas & tasks as a standard list of strings
eval_arenas:

  - arena: magpie/multi_longsleeve_provide_semkey_pixel_no_success_stop_resol_128_workspace_height_2
    task: magpie/canonicalisation_alignment

  - arena: magpie/multi_longsleeve_provide_semkey_pixel_no_success_stop_resol_128_workspace
    task: magpie/canonicalisation_alignment
  
  - arena: magpie/multi_longsleeve_provide_semkey_pixel_no_success_stop_resol_128_workspace
    task: magpie/central_alignment
  
  - arena: magpie/multi_dress_provide_semkey_pixel_no_success_stop_resol_128_workspace
    task: magpie/central_alignment
  
  - arena: magpie/multi_skirt_provide_semkey_pixel_no_success_stop_resol_128_workspace
    task: magpie/central_alignment

  - arena: magpie/multi_trousers_provide_semkey_pixel_no_success_stop_resol_128_workspace
    task: magpie/central_alignment
EOF

echo "✅ Created: $TRANSFER_EVAL_FILE"