#!/bin/bash

# Submit the recovered LaGarNet diffusion-policy baselines (train + eval) to Viking in one go.
# These configs run the diffusion policy via Magpie's DDPM mode (loss_type: diffusion,
# primitive_integration: none) — single-picker single-primitive flattening, one policy per
# garment, at two demo budgets (50 and 200). They use the plain train+eval pipeline
# (train_and_evaluate_single via hydra_train.py), so NO -a/-r transfer flag.
#
# Usage:
#   ./job_scripts/submit_lagarnet_diffusion_jobs.sh            # submit everything
#   ./job_scripts/submit_lagarnet_diffusion_jobs.sh --dry-run  # print the commands only

DRY_RUN=false
if [ "$1" == "--dry-run" ]; then DRY_RUN=true; fi

# Slurm resource request (per job). Adjust here if needed.
CPUS=6
MEM="24G"
TIME="48:00:00"

GARMENTS=(dress longsleeve trousers skirt)
DEMOS=(50 200)

EXPS=()
for g in "${GARMENTS[@]}"; do
    for d in "${DEMOS[@]}"; do
        EXPS+=("diffusion_single_picker_single_primitive_multi_${g}_flattening_demo_${d}")
    done
done

i=0
for exp in "${EXPS[@]}"; do
    i=$((i + 1))
    # Every 5th job -> gpuplus (20%); the rest -> gpu (80%).
    if [ $((i % 5)) -eq 0 ]; then
        PARTITION="gpuplus"
    else
        PARTITION="gpu"
    fi

    CMD="./job_scripts/generate_and_submit_viking_job.sh lagarnet/${exp} -c ${CPUS} -m ${MEM} -p ${PARTITION} -t ${TIME}"
    echo "[submit_lagarnet_diffusion_jobs] (${i}/${#EXPS[@]}) [${PARTITION}] ${exp}"
    if [ "$DRY_RUN" = false ]; then
        $CMD
    else
        echo "    $CMD"
    fi
done
