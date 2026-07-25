#!/bin/bash

# Submit the v158+ MAGPIE sim-to-real sweep.
#
# Motivation: MAGPIE reaches ~75% success in simulation but transfers poorly to the real
# dual-arm rig, including the sim+real "mix" agents. This sweep isolates the four suspected
# causes, all of which are visible in the code rather than the results:
#
#   1. Real data is drowned. 39 real trajectories against 800 sim ones under plain
#      shuffling puts real at ~3-5% of a 1024-sample batch, and the `domain` channel the
#      combine script writes was never read. -> Group C (domain_oversample_ratio).
#   2. `random_swap_actions` trains gripper order as interchangeable. True in sim, false on
#      a limited-reach dual-arm rig, and it invites midpoint-collapsed grasps. -> v160/v161.
#   3. Rotation is uniform over the FULL 360 degrees, clamps out-of-frame targets onto the
#      image border, and (via `not_rotate_primitives: [0]`) lets scene orientation leak the
#      primitive identity. -> v158 (none) / v159 (bounded +/-20 deg).
#   4. Colour is a nuisance dimension the real camera does not reproduce. -> Group B (grayscale).
#
# Base recipe for every version: v150 (= v126 + prim_weight 0.003), the strongest complete
# run of the previous batch.
#
# Stages
#   1  Groups A/B/C: train + 6-way zero-shot transfer eval (-a). Submit these first.
#   2  Group D: sim-pretrain -> real-finetune. Needs stage-1 checkpoints, so only submit
#      once the corresponding stage-1 runs have finished.
#
# Usage:
#   ./job_scripts/submit_new_magpie_jobs.sh                    # stage 1 (default)
#   ./job_scripts/submit_new_magpie_jobs.sh --stage 2          # fine-tune runs
#   ./job_scripts/submit_new_magpie_jobs.sh --stage all        # both
#   ./job_scripts/submit_new_magpie_jobs.sh --offline-eval     # offline real-action scoring
#   ./job_scripts/submit_new_magpie_jobs.sh --dry-run          # print commands only
#
# 80% of training jobs go to the `gpu` partition, 20% to `gpuplus`
# (deterministic: every 5th submission goes to gpuplus).

DRY_RUN=false
STAGE="1"
OFFLINE_EVAL=false

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)      DRY_RUN=true; shift ;;
        --offline-eval) OFFLINE_EVAL=true; shift ;;
        --stage)        STAGE="$2"; shift 2 ;;
        --stage=*)      STAGE="${1#*=}"; shift ;;
        -h|--help)      sed -n '3,36p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# --- Stage 1 -----------------------------------------------------------------

# Group A: augmentation ablations on the all-sim dataset.
STAGE1_EXPS=(
    magpie/magpie_ctr_align_all_sim_garments_p4_v158_hindsight   # no rotation
    magpie/magpie_ctr_align_all_sim_garments_p4_v159_hindsight   # rotation bounded to +/-20 deg
    magpie/magpie_ctr_align_all_sim_garments_p4_v160_hindsight   # no action swap
    magpie/magpie_ctr_align_all_sim_garments_p4_v161_hindsight   # conservative (bounded rot + no swap + no scale)

    # Group B: grayscale input (2 encoder channels).
    magpie/magpie_ctr_align_all_sim_garments_p4_v162_hindsight   # grayscale, standard augmentation
    magpie/magpie_ctr_align_all_sim_garments_p4_v163_hindsight   # grayscale + conservative

    # Group C: sim+real mix. The domain-rebalanced runs are the direct test of hypothesis 1.
    magpie/magpie_ctr_align_mix_sim_and_real_garments_p4_v164_hindsight   # real 25% of each batch
    magpie/magpie_ctr_align_mix_sim_and_real_garments_p4_v165_hindsight   # real 50% of each batch
    magpie/magpie_ctr_align_mix_sim_and_real_garments_p4_v166_hindsight   # real 50% + conservative
    magpie/magpie_ctr_align_mix_sim_and_real_garments_p4_v167_hindsight   # real 50% + grayscale + conservative
    magpie/magpie_ctr_align_mix_sim_and_real_garments_p4_v158_hindsight   # mix twin, no rotation
    magpie/magpie_ctr_align_mix_sim_and_real_garments_p4_v161_hindsight   # mix twin, conservative
    magpie/magpie_ctr_align_mix_sim_and_real_garments_p4_v163_hindsight   # mix twin, grayscale + conservative
)

# --- Stage 2: sim-pretrain -> real-finetune ----------------------------------
# Each entry initialises from a stage-1 checkpoint via `init_from_exp`.
STAGE2_EXPS=(
    magpie/magpie_ctr_align_ft_real_only_p4_v168_hindsight   # from v161, real-only
    magpie/magpie_ctr_align_ft_mix_p4_v169_hindsight         # from v161, mix with real at 50%
    magpie/magpie_ctr_align_ft_real_only_p4_v170_hindsight   # from v163 (grayscale), real-only
    magpie/magpie_ctr_align_ft_mix_p4_v171_hindsight         # from v163 (grayscale), mix at 50%
)

# Stage-1 experiments the stage-2 runs depend on.
STAGE2_DEPENDENCIES=(
    magpie/magpie_ctr_align_all_sim_garments_p4_v161_hindsight
    magpie/magpie_ctr_align_all_sim_garments_p4_v163_hindsight
)

# --- Offline real-action eval ------------------------------------------------
# Scores inferred actions against the recorded human actions in the real dataset. Includes
# the previous batch's leaders so the notebook can compare old and new agents on one axis.
OFFLINE_BASELINES=(
    magpie/magpie_ctr_align_all_sim_garments_p4_v126_hindsight
    magpie/magpie_ctr_align_all_sim_garments_p4_v140_hindsight
    magpie/magpie_ctr_align_all_sim_garments_p4_v150_hindsight
    magpie/magpie_ctr_align_all_sim_garments_p4_v157_hindsight
    magpie/magpie_ctr_align_mix_sim_and_real_garments_p4_v140_hindsight
    magpie/magpie_ctr_align_mix_sim_and_real_garments_p4_v145_hindsight
    magpie/magpie_ctr_align_mix_sim_and_real_garments_p4_v157_hindsight
)

# --- Submission helpers ------------------------------------------------------

submit_index=0

submit_training() {
    local exp="$1"
    submit_index=$((submit_index + 1))

    # Every 5th job -> gpuplus (20%); the rest -> gpu (80%).
    local partition="gpu"
    if [ $((submit_index % 5)) -eq 0 ]; then
        partition="gpuplus"
    fi

    local cmd="./job_scripts/generate_and_submit_viking_job.sh ${exp} -c 6 -m 24G -p ${partition} -t 58:00:00 -a"
    echo "[submit_new_magpie_jobs] [train:${partition}] ${exp}"
    if [ "$DRY_RUN" = false ]; then $cmd; else echo "    $cmd"; fi
}

submit_offline_eval() {
    local exp="$1"
    local cmd="./job_scripts/generate_and_submit_viking_job.sh ${exp} -c 4 -m 16G -p gpu_short -o"
    echo "[submit_new_magpie_jobs] [offline-eval] ${exp}"
    if [ "$DRY_RUN" = false ]; then $cmd; else echo "    $cmd"; fi
}

# --- Main --------------------------------------------------------------------

if [ "$OFFLINE_EVAL" = true ]; then
    ALL_OFFLINE=("${STAGE1_EXPS[@]}" "${STAGE2_EXPS[@]}" "${OFFLINE_BASELINES[@]}")
    echo "[submit_new_magpie_jobs] Submitting ${#ALL_OFFLINE[@]} offline action-prediction evals."
    echo "[submit_new_magpie_jobs] Agents without a checkpoint yet will fail fast; that is expected."
    for exp in "${ALL_OFFLINE[@]}"; do
        submit_offline_eval "$exp"
    done
    exit 0
fi

case "$STAGE" in
    1)   EXPS=("${STAGE1_EXPS[@]}") ;;
    2)   EXPS=("${STAGE2_EXPS[@]}") ;;
    all) EXPS=("${STAGE1_EXPS[@]}" "${STAGE2_EXPS[@]}") ;;
    *)   echo "Invalid --stage '$STAGE' (expected 1, 2 or all)" >&2; exit 1 ;;
esac

if [ "$STAGE" = "2" ] || [ "$STAGE" = "all" ]; then
    echo "[submit_new_magpie_jobs] NOTE: the stage-2 fine-tune runs load"
    for dep in "${STAGE2_DEPENDENCIES[@]}"; do
        echo "    <save_root>/${dep##*/}/checkpoints/net_best.pt"
    done
    echo "  They will abort immediately if those stage-1 runs have not finished."
    if [ "$STAGE" = "all" ]; then
        echo "  With --stage all they are queued alongside stage 1 and WILL fail; prefer"
        echo "  running --stage 1 now and --stage 2 once stage 1 completes."
    fi
    echo
fi

echo "[submit_new_magpie_jobs] Stage ${STAGE}: submitting ${#EXPS[@]} jobs."
for exp in "${EXPS[@]}"; do
    submit_training "$exp"
done
