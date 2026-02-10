#!/bin/bash

# 1. Check if a config name was provided
if [ -z "$1" ]; then
    echo "Usage: ./submit_evaluating_locally.sh <config_name> [f]"
    echo "  f : Run in foreground (no log file)"
    exit 1
fi

CONFIG_NAME=$1
MODE=$2
LOG_DIR="tmp"

if [ "$MODE" == "f" ]; then
    echo "--- Running in FOREGROUND ---"
    python -u tool/hydra_train.py --config-name "run_exp/$CONFIG_NAME"
else
    mkdir -p "$LOG_DIR"
    echo "--- Running in BACKGROUND ---"

    # 2. Use a wrapper to ensure the log file name includes the correct PID
    # We launch a background task that captures its own PID to name the file
    (
        # Launch python in background inside this subshell
        python -u tool/hydra_eval.py --config-name "run_exp/$CONFIG_NAME" &
        PYTHON_PID=$!
        
        # Create a symlink or move the output to a file named with the PID
        # But for simplicity, we redirect the outer subshell to a file 
        # based on the PID we just captured.
        echo $PYTHON_PID > "$LOG_DIR/last_pid.txt"
        wait $PYTHON_PID
    ) > "$LOG_DIR/${CONFIG_NAME}_TEMP.txt" 2>&1 &
    
    # 3. Give the subshell a moment to write the PID file
    sleep 0.5
    ACTUAL_PID=$(cat "$LOG_DIR/last_pid.txt")
    mv "$LOG_DIR/${CONFIG_NAME}_TEMP.txt" "$LOG_DIR/${CONFIG_NAME}_${ACTUAL_PID}.txt"
    rm "$LOG_DIR/last_pid.txt"

    # 4. Disown the background wrapper
    disown $!
    
    echo "Log file: $LOG_DIR/${CONFIG_NAME}_${ACTUAL_PID}.txt"
    echo "Process started with PID: $ACTUAL_PID."
fi