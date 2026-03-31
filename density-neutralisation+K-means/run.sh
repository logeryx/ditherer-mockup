#!/bin/bash

# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================
COLORS=4
SCALE=6
BLEND=0.3

# THE MAGIC PARAMETER:
# 32 or 48 will heavily compress the shadows into a single gray/green 
# and easily dedicate 2 whole palette slots to your torch.
BIN_SIZE=12

# Pointing to the new stable binned script
PYTHON_SCRIPT="density-neutralisation+K-means/scripts/dither.py"
# ==============================================================================

# 1. Validate Input
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input_video_file>"
    echo "Example: $0 my_gameplay.mp4"
    exit 1
fi

INPUT_VIDEO="$1"

if [ ! -f "$INPUT_VIDEO" ]; then
    echo "Error: Input file '$INPUT_VIDEO' does not exist."
    exit 1
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found in the current directory."
    exit 1
fi

# 2. Extract Base Filename
BASENAME=$(basename -- "$INPUT_VIDEO")
BASENAME_NO_EXT="${BASENAME%.*}"

# 3. Create a Unique Working Directory
WORK_DIR="workspace_${BASENAME_NO_EXT}"
COUNTER=1
while [ -d "$WORK_DIR" ]; do
    WORK_DIR="workspace_${BASENAME_NO_EXT}_${COUNTER}"
    COUNTER=$((COUNTER + 1))
done

DIR_IN="$WORK_DIR/frames_in"
DIR_OUT="$WORK_DIR/frames_out"

mkdir -p "$DIR_IN"
mkdir -p "$DIR_OUT"

echo "---------------------------------------------------------"
echo " Started Stable Binned Pipeline for: $BASENAME"
echo " Working Directory: $WORK_DIR"
echo " Settings: $COLORS Colors, ${SCALE}x Scale, $BLEND Blend"
echo " Bin Size: $BIN_SIZE"
echo "---------------------------------------------------------"

# 4. Detect Framerate
echo "[1/4] Detecting framerate..."
FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 "$INPUT_VIDEO")

if [ -z "$FPS" ]; then
    echo "      Warning: Could not detect framerate. Defaulting to 30."
    FPS="30"
else
    echo "      Detected framerate: $FPS"
fi

# 5. Extract Frames
echo "[2/4] Extracting frames..."
ffmpeg -v warning -stats -i "$INPUT_VIDEO" -q:v 2 "$DIR_IN/frame_%08d.jpg"

if [ $? -ne 0 ]; then
    echo "Error: FFmpeg frame extraction failed!"
    exit 1
fi

# 6. Apply Python Dithering
echo "[3/4] Applying dithering..."
.venv/bin/python "$PYTHON_SCRIPT" "$DIR_IN" "$DIR_OUT" -c $COLORS -s $SCALE --blend $BLEND -b $BIN_SIZE

if [ $? -ne 0 ]; then
    echo "Error: Python dithering script failed!"
    exit 1
fi

# 7. Reassemble Video
echo "[4/4] Reassembling output video..."
OUTPUT_VIDEO="${BASENAME_NO_EXT}_stable_binned_c${COLORS}_b${BLEND}_b${BIN_SIZE}.mp4"

ffmpeg -v warning -stats -framerate "$FPS" -i "$DIR_OUT/frame_%08d-dither.png" -c:v libx264 -pix_fmt yuv420p -crf 18 "$OUTPUT_VIDEO"

if [ $? -ne 0 ]; then
    echo "Error: FFmpeg video reassembly failed!"
    exit 1
fi

echo "---------------------------------------------------------"
echo " Success! Final video saved as:"
echo " -> $OUTPUT_VIDEO"
echo "---------------------------------------------------------"