#!/bin/bash
# Run MAPPO baselines on all environments with the world_state fix
# Version 5: Full training with 100M-150M timesteps per environment
# Usage: bash scripts/run_mappo_all_v5.sh [gpu]
#
# Progress: cat mappo_status_v5.txt

set -uo pipefail

GPU="${1:-0}"
LOGDIR="run_mappo_logs_v5"
STATUS_FILE="mappo_status_v5.txt"
RESULTSFile="run_mappo_logs_v5/mappo_results.tsv"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# IPPO baselines for comparison
declare -A IPPO_BASELINE=(
    ["coin_game"]="18.807"
    ["harvest_common_open"]="115.153"
    ["clean_up"]="0.0"
    ["pd_arena"]="15.550"
    ["coop_mining"]="185.812"
    ["territory_open"]="195.240"
    ["mushrooms"]="224.079"
    ["gift"]="101.391"
)

mkdir -p "$LOGDIR"

# Environment list
ENVS=(
    coin_game
    harvest_common_open
    clean_up
    pd_arena
    coop_mining
    territory_open
    mushrooms
    gift
)

# Timesteps per environment (based on CF hyperparameters)
declare -A ENV_TIMESTEPS=(
    ["coin_game"]="150000000"
    ["harvest_common_open"]="100000000"
    ["clean_up"]="100000000"
    ["pd_arena"]="150000000"
    ["coop_mining"]="150000000"
    ["territory_open"]="150000000"
    ["mushrooms"]="100000000"
    ["gift"]="150000000"
)

# Fixed hyperparameters for all environments
NUM_ENVS=128
NUM_STEPS=128
UPDATE_EPOCHS=4
NUM_MINIBATCHES=32

echo "========================================"
echo "  MAPPO Baselines v5 (Full Training)"
echo "========================================"
echo "GPU: $GPU"
echo "Log dir: $LOGDIR"
echo "Results: $RESULTSFile"
echo "Timesteps: 100M-150M per environment"
echo "========================================"
echo ""

# Results file header
echo -e "env\tmappo_return\tippo_baseline\tdiff\tstatus\ttimesteps" > "$RESULTSFile"

START_TIME=$(date +%s)
NUM_OK=0
NUM_CRASH=0

for i in "${!ENVS[@]}"; do
    ENV="${ENVS[$i]}"
    ENV_IDX=$((i + 1))
    LOGFILE="$LOGDIR/${ENV}.log"
    TIMESTEPS="${ENV_TIMESTEPS[$ENV]}"

    IPPO_VAL="${IPPO_BASELINE[$ENV]}"

    elapsed=$(( $(date +%s) - START_TIME ))

    echo "--- Running $ENV ($ENV_IDX/${#ENVS[@]}, timesteps=$TIMESTEPS) ---"

    # Update status
    cat > "$STATUS_FILE" <<EOF
=== MAPPO Baselines v5 (Full Training) ===
updated:   $(date '+%Y-%m-%d %H:%M:%S')
gpu:       $GPU
elapsed:   ${elapsed}s ($(( elapsed / 60 ))m)

progress:  $ENV_IDX/${#ENVS[@]} — training $ENV (timesteps=$TIMESTEPS)
IPPO baseline: $IPPO_VAL
EOF

    CUDA_VISIBLE_DEVICES="$GPU" \
    /home/shuqing/.conda/envs/melting-jax/bin/python -u scripts/train.py \
        --algorithm mappo \
        --env "$ENV" \
        --timesteps "$TIMESTEPS" \
        --seed 42 \
        --num-envs "$NUM_ENVS" \
        --num-steps "$NUM_STEPS" \
        --lr 5e-4 \
        --update-epochs "$UPDATE_EPOCHS" \
        --num-minibatches "$NUM_MINIBATCHES" \
        > "$LOGFILE" 2>&1
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "  $ENV: CRASH (exit $EXIT_CODE)"
        echo -e "$ENV\t0.000\t$IPPO_VAL\t-$IPPO_VAL\tcrash\t$TIMESTEPS" >> "$RESULTSFile"
        NUM_CRASH=$((NUM_CRASH + 1))
    else
        # Extract final return from log
        RETURN_LINE=$(grep "return=" "$LOGFILE" | tail -1 || true)
        if [ -z "$RETURN_LINE" ]; then
            echo "  $ENV: No return found"
            RETURN_VAL="0.000"
            STATUS="no_return"
        else
            RETURN_VAL=$(echo "$RETURN_LINE" | grep -oP 'return=\K[0-9.-]+')
            echo "  $ENV: return=$RETURN_VAL"
            STATUS="ok"
            NUM_OK=$((NUM_OK + 1))
        fi

        # Calculate diff
        DIFF=$(echo "$RETURN_VAL - $IPPO_VAL" | bc)
        echo -e "$ENV\t$RETURN_VAL\t$IPPO_VAL\t$DIFF\t$STATUS\t$TIMESTEPS" >> "$RESULTSFile"
    fi
done

    # Final status
    elapsed=$(( $(date +%s) - START_TIME ))
    cat > "$STATUS_FILE" <<EOF
=== MAPPO Baselines v5 (Full Training) ===
updated:   $(date '+%Y-%m-%d %H:%M:%S')
gpu:       $GPU
elapsed:   ${elapsed}s ($(( elapsed / 3600 ))h)

results:   see $RESULTSFile
ok:        $NUM_OK / ${#ENVS[@]}
crashes:   $NUM_CRASH
EOF

echo ""
echo "========================================"
echo "  RESULTS SUMMARY"
echo "========================================"
column -t -s $'\t' "$RESULTSFile"
echo ""
echo "Ok: $NUM_OK / ${#ENVS[@]} | Crashes: $NUM_CRASH"
echo "Total time: $(( elapsed / 3600 ))h $(( (elapsed % 3600) / 60 ))m"
echo "Results saved to: $RESULTSFile"
echo "Logs saved to: $LOGDIR"
echo "========================================"
