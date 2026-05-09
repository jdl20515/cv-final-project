#!/usr/bin/env bash
# Download Vimeo-90K septuplet dataset
# http://toflow.csail.mit.edu/

set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/../data" && pwd)"
mkdir -p "$DATA_DIR"

ARCHIVE="vimeo_septuplet.zip"
URL="http://data.csail.mit.edu/tofu/dataset/$ARCHIVE"

if [ -d "$DATA_DIR/vimeo_septuplet" ]; then
    echo "Vimeo-90K septuplet already exists at $DATA_DIR/vimeo_septuplet"
    exit 0
fi

echo "Downloading Vimeo-90K septuplet dataset..."
echo "This is a large download (~82GB). Make sure you have enough disk space."

cd "$DATA_DIR"

if [ ! -f "$ARCHIVE" ]; then
    curl -L -O "$URL"
fi

echo "Extracting..."
unzip -q "$ARCHIVE"

echo "Downloading train/test split lists..."
curl -L -O "http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet/sep_trainlist.txt" \
    -o "vimeo_septuplet/sep_trainlist.txt"
curl -L -O "http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet/sep_testlist.txt" \
    -o "vimeo_septuplet/sep_testlist.txt"

echo "Done! Dataset is at: $DATA_DIR/vimeo_septuplet"
