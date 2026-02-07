#!/bin/bash

# 1. Check if a config name was provided
if [ -z "$1" ]; then
    echo "Usage: ./submit_running_locally.sh <config_name> [f]"
    echo "  f : Run in foreground (optional)"
    exit 1
fi

CONFIG_NAME=$1
MODE=$2 # Capture the second argument
LOG_DIR="tmp"
LOG_FILE="$LOG_DIR/${CONFIG_NAME}.txt"

# 2. Ensure the log directory exists
mkdir -p "$LOG_DIR"

echo "Experiment: $CONFIG_NAME"

# 3. Execution Logic
if [ "$MODE" == "f" ]; then
    echo "Running in FOREGROUND... (Press Ctrl+C to stop)"
    # Run directly in terminal, still logging to file and console via 'tee'
    python -u tool/hydra_train.py --config-name "run_exp/$CONFIG_NAME" 2>&1 | tee "$LOG_FILE"
else
    echo "Running in BACKGROUND..."
    echo "Logging output to: $LOG_FILE"
    
    # Run in background
    python -u tool/hydra_train.py --config-name "run_exp/$CONFIG_NAME" > "$LOG_FILE" 2>&1 &
    
    PID=$!
    disown $PID
    echo "Process started with PID: $PID. You can now safely close this terminal."
fi