#!/bin/bash

# 1. Check if a config name was provided
if [ -z "$1" ]; then
    echo "Usage: ./submit_running_locally.sh <config_name>"
    exit 1
fi

CONFIG_NAME=$1
LOG_DIR="tmp"
LOG_FILE="$LOG_DIR/${CONFIG_NAME}.txt"

# 2. Ensure the log directory exists
mkdir -p "$LOG_DIR"

echo "Starting experiment: $CONFIG_NAME"
echo "Logging output to: $LOG_FILE"

# 3. Run the command in the background
# We use -u for unbuffered output to ensure the log file updates in real-time
python -u tool/hydra_train.py --config-name "run_exp/$CONFIG_NAME" > "$LOG_FILE" 2>&1 &

# 4. Get the Process ID (PID) of the last background command
PID=$!

# 5. Disown the process so it persists after the terminal closes
disown $PID

echo "Process started with PID: $PID. You can now safely close this terminal."