#!/bin/bash

# Usage: ./compress_gif.sh <input.gif> <output.gif> [speed_factor]
# Example: ./compress_gif.sh in.gif out.gif 0.5 (for slow motion)
# Example: ./compress_gif.sh in.gif out.gif 3   (for 3x speed)

INPUT=$1
OUTPUT=$2
SPEED=${3:-2} # Defaults to 2 if $3 is not provided

if [ -z "$INPUT" ] || [ -z "$OUTPUT" ]; then
    echo "Usage: $0 <input.gif> <output.gif> [speed_factor]"
    exit 1
fi

# 1. Get duration
DURATION=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$INPUT")
TARGET_MAX_FRAMES=1000

# 2. Calculate New Duration
# Speed > 1 makes it shorter, Speed < 1 makes it longer (slow-mo)
NEW_DURATION=$(echo "scale=4; $DURATION / $SPEED" | bc)

# 3. Calculate FPS to stay under 1000 frames
# Formula: FPS = 1000 / New_Duration
CALC_FPS=$(echo "scale=2; $TARGET_MAX_FRAMES / $NEW_DURATION" | bc)

# 4. Safety Cap
# We cap at 20 FPS for size, but only if the calculation allows it.
if (( $(echo "$CALC_FPS > 20" | bc -l) )); then
    FINAL_FPS=20
else
    FINAL_FPS=$CALC_FPS
fi

echo "--- Processing Settings ---"
echo "Speed Factor:  ${SPEED}x"
echo "New Duration:  ${NEW_DURATION}s"
echo "Target FPS:    ${FINAL_FPS}"
echo "---------------------------"

# 5. Run Conversion
# Use '1/SPEED' for setpts logic (e.g., 2x speed needs setpts=0.5*PTS)
PTS_FACTOR=$(echo "scale=4; 1 / $SPEED" | bc)
FILTER="setpts=${PTS_FACTOR}*PTS,fps=$FINAL_FPS,scale='min(480,iw)':-1:flags=lanczos"

ffmpeg -i "$INPUT" -vf "$FILTER,palettegen" -y palette.png
ffmpeg -i "$INPUT" -i palette.png -t "$NEW_DURATION" -filter_complex "${FILTER}[x];[x][1:v]paletteuse=dither=floyd_steinberg" -y "$OUTPUT"

rm palette.png

# 6. Verify result
ACTUAL_FRAMES=$(ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 "$OUTPUT")
echo "Done! Output has $ACTUAL_FRAMES frames."