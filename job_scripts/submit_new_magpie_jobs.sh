#!/bin/bash

# Submit the v140+ MAGPIE sweep (train + 6-way zero-shot transfer eval, -a) in one go.
#   - sim-and-real (mix) configurations are submitted first
#   - 80% of jobs go to the `gpu` partition, 20% to `gpuplus`
#     (deterministic: every 5th submission goes to gpuplus)
#
# Usage:
#   ./job_scripts/submit_new_magpie_jobs.sh            # submit everything
#   ./job_scripts/submit_new_magpie_jobs.sh --dry-run  # print the commands only

DRY_RUN=false
if [ "$1" == "--dry-run" ]; then DRY_RUN=true; fi

# Mix (sim + real garments) configurations — submitted first.
MIX_EXPS=(
    magpie_ctr_align_mix_sim_and_real_garments_p4_v140_hindsight
    magpie_ctr_align_mix_sim_and_real_garments_p4_v141_hindsight
    magpie_ctr_align_mix_sim_and_real_garments_p4_v143_hindsight
    magpie_ctr_align_mix_sim_and_real_garments_p4_v145_hindsight
    magpie_ctr_align_mix_sim_and_real_garments_p4_v149_hindsight
    magpie_ctr_align_mix_sim_and_real_garments_p4_v151_hindsight
    magpie_ctr_align_mix_sim_and_real_garments_p4_v155_hindsight
    magpie_ctr_align_mix_sim_and_real_garments_p4_v156_hindsight
    magpie_ctr_align_mix_sim_and_real_garments_p4_v157_hindsight
)

# All-sim configurations.
SIM_EXPS=(
    magpie_ctr_align_all_sim_garments_p4_v140_hindsight
    magpie_ctr_align_all_sim_garments_p4_v141_hindsight
    magpie_ctr_align_all_sim_garments_p4_v142_hindsight
    magpie_ctr_align_all_sim_garments_p4_v143_hindsight
    magpie_ctr_align_all_sim_garments_p4_v144_hindsight
    magpie_ctr_align_all_sim_garments_p4_v145_hindsight
    magpie_ctr_align_all_sim_garments_p4_v146_hindsight
    magpie_ctr_align_all_sim_garments_p4_v147_hindsight
    magpie_ctr_align_all_sim_garments_p4_v148_hindsight
    magpie_ctr_align_all_sim_garments_p4_v149_hindsight
    magpie_ctr_align_all_sim_garments_p4_v150_hindsight
    magpie_ctr_align_all_sim_garments_p4_v151_hindsight
    magpie_ctr_align_all_sim_garments_p4_v152_hindsight
    magpie_ctr_align_all_sim_garments_p4_v153_hindsight
    magpie_ctr_align_all_sim_garments_p4_v154_hindsight
    magpie_ctr_align_all_sim_garments_p4_v155_hindsight
    magpie_ctr_align_all_sim_garments_p4_v156_hindsight
    magpie_ctr_align_all_sim_garments_p4_v157_hindsight
)

ALL_EXPS=("${MIX_EXPS[@]}" "${SIM_EXPS[@]}")

i=0
for exp in "${ALL_EXPS[@]}"; do
    i=$((i + 1))
    # Every 5th job -> gpuplus (20%); the rest -> gpu (80%).
    if [ $((i % 5)) -eq 0 ]; then
        PARTITION="gpuplus"
    else
        PARTITION="gpu"
    fi

    CMD="./job_scripts/generate_and_submit_viking_job.sh magpie/${exp} -c 6 -m 24G -p ${PARTITION} -t 58:00:00 -a"
    echo "[submit_new_magpie_jobs] (${i}/${#ALL_EXPS[@]}) [${PARTITION}] ${exp}"
    if [ "$DRY_RUN" = false ]; then
        $CMD
    else
        echo "    $CMD"
    fi
done
