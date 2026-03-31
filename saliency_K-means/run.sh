#!/bin/bash

# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================
COLORS=5
SCALE=6
BLEND=0.05

# --- Saliency & Luminance Constants ---
EDGE_WEIGHT=15.0
SAMPLES=5000

# NEW: How much more important is brightness than hue? (e.g., 2.5x more important)
L_WEIGHT=2.5

# Pointing to the new saliency script
PYTHON_SCRIPT="scripts/saliency_kmeans_dither.py"
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
echo " Started Saliency/CIELAB Pipeline for: $BASENAME"
echo " Working Directory: $WORK_DIR"
echo " Settings: $COLORS Colors, ${SCALE}x Scale, $BLEND Blend"
echo " Optics: ${L_WEIGHT}x L-Weight, ${EDGE_WEIGHT}x Extremes"
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
.venv/bin/python "$PYTHON_SCRIPT" "$DIR_IN" "$DIR_OUT" -c $COLORS -s $SCALE --blend $BLEND --edge-weight $EDGE_WEIGHT --samples $SAMPLES --l-weight $L_WEIGHT

if [ $? -ne 0 ]; then
    echo "Error: Python dithering script failed!"
    exit 1
fi

# 7. Reassemble Video
echo "[4/4] Reassembling output video..."
OUTPUT_VIDEO="${BASENAME_NO_EXT}_saliency_c${COLORS}_s${SCALE}_L${L_WEIGHT}.mp4"

ffmpeg -v warning -stats -framerate "$FPS" -i "$DIR_OUT/frame_%08d-dither.png" -c:v libx264 -pix_fmt yuv420p -crf 18 "$OUTPUT_VIDEO"

if [ $? -ne 0 ]; then
    echo "Error: FFmpeg video reassembly failed!"
    exit 1
fi

echo "---------------------------------------------------------"
echo " Success! Final video saved as:"
echo " -> $OUTPUT_VIDEO"
echo "---------------------------------------------------------"