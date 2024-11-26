#!/bin/bash

# Define source and destination base directories
SOURCE_BASE_DIR=~/audio-diffusion/cache/spotify_sleep_dataset/
DEST_BASE_DIR=th716@login.hpc.cam.ac.uk:/home/th716/rds/hpc-work/audio-diffusion/cache/spotify_sleep_dataset/

# Loop through all directories in the source base directory
for dir in $SOURCE_BASE_DIR/*; do
  if [ -d "$dir" ]; then
    # Extract the directory name
    DIR_NAME=$(basename "$dir")

    # Check if the directory name matches the pattern 'mel_spec_{int}_{int}'
    if [[ $DIR_NAME =~ ^mel_spec_[0-9]+_[0-9]+$ ]]; then
      # Print the directory name
      echo "Transferring directory: $DIR_NAME"
      
      # Form the rsync command
      rsync -av "$dir"/ "$DEST_BASE_DIR/$DIR_NAME"/ --progress --delete
    fi
  fi
done
