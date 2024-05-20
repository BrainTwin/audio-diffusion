#!/bin/bash

# Array of model names
model_names=(
    "clap-laion-audio" "clap-laion-music" "vggish" 
    "MERT-v1-95M" "clap-2023" "encodec-emb" "encodec-emb-48k"
)

# Array of directory pairs
dir_pairs=(
    "cache/drum_samples/waveform cache/fma_pop/waveform"
    "cache/musiccaps/waveform cache/spotify_sleep_dataset/waveform"
)

# Function to extract the last two elements of a path
get_last_two_paths() {
    echo "$1" | awk -F'/' '{print $(NF-1) "/" $NF}'
}

# Iterate over each model name
for model_name in "${model_names[@]}"; do
    # Iterate over each directory pair
    for dir_pair in "${dir_pairs[@]}"; do
        # Split the directory pair into two separate variables
        dir1=$(echo $dir_pair | awk '{print $1}')
        dir2=$(echo $dir_pair | awk '{print $2}')
        # Extract the last two elements of each path
        dir1_last_two=$(get_last_two_paths $dir1)
        dir2_last_two=$(get_last_two_paths $dir2)
        # Create the log file name
        log_file="logs/calculating_embeddings_${model_name}_${dir1_last_two//\//_}_${dir2_last_two//\//_}.log"
        # Run the command
        nohup bash -c "fadtk \"$model_name\" \"$dir1\" \"$dir2\" > \"$log_file\" 2>&1" &
        wait $! # Wait for the current command to finish
    done
done
