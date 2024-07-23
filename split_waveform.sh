#!/bin/bash

# Define the source and target directories
SOURCE_DIR="cache/fma_pop/waveform"
TARGET_DIR1="cache/fma_pop/waveform_half_1"
TARGET_DIR2="cache/fma_pop/waveform_half_2"

# Create the target directories if they don't exist
mkdir -p "$TARGET_DIR1"
mkdir -p "$TARGET_DIR2"

# Initialize a counter
count=1

# Loop through all .wav files in the source directory
for file in "$SOURCE_DIR"/*.mp3; do
  if [ $((count % 2)) -eq 1 ]; then
    # Odd numbered files go to waveform_half_1
    cp "$file" "$TARGET_DIR1"
  else
    # Even numbered files go to waveform_half_2
    cp "$file" "$TARGET_DIR2"
  fi
  # Increment the counter
  count=$((count + 1))
done

echo "Files have been copied successfully."
