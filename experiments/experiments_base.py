import os

# List of variable combinations
variable_combinations = [
    {
        "dataset_name": "/home/th716/rds/hpc-work/audio-diffusion/cache/musiccaps/mel_spec_64_64",
        "output_dir": "/home/th716/rds/hpc-work/audio-diffusion/models/musiccaps_64_64"
    },
    {
        "dataset_name": "/home/th716/rds/hpc-work/audio-diffusion/cache/drum_samples/mel_spec_64_64",
        "output_dir": "/home/th716/rds/hpc-work/audio-diffusion/models/drum_samples_64_64"
    },
    {
        "dataset_name": "/home/th716/rds/hpc-work/audio-diffusion/cache/spotify_sleep_dataset/mel_spec_64_64",
        "output_dir": "/home/th716/rds/hpc-work/audio-diffusion/models/spotify_sleep_dataset_64_64"
    },
    {
        "dataset_name": "/home/th716/rds/hpc-work/audio-diffusion/cache/musiccaps/mel_spec_256_256",
        "output_dir": "/home/th716/rds/hpc-work/audio-diffusion/models/musiccaps_256_256"
    },
    {
        "dataset_name": "/home/th716/rds/hpc-work/audio-diffusion/cache/drum_samples/mel_spec_256_256",
        "output_dir": "/home/th716/rds/hpc-work/audio-diffusion/models/drum_samples_256_256"
    },
    {
        "dataset_name": "/home/th716/rds/hpc-work/audio-diffusion/cache/spotify_sleep_dataset/mel_spec_256_256",
        "output_dir": "/home/th716/rds/hpc-work/audio-diffusion/models/spotify_sleep_dataset_256_256"
    }
]

# Loop over the variable combinations
for vars in variable_combinations:
    command = f"""
    accelerate launch --config_file config/accelerate_local.yaml \\
    scripts/train_unet.py \\
    --dataset_name {vars['dataset_name']} \\
    --hop_length 256 \\
    --n_fft 1024 \\
    --output_dir {vars['output_dir']} \\
    --train_batch_size 32 \\
    --eval_batch_size 16 \\
    --num_epochs 1000 \\
    --max_training_num_steps 300000 \\
    --gradient_accumulation_steps 4 \\
    --learning_rate 1e-4 \\
    --lr_warmup_steps 500 \\
    --mixed_precision no \\
    --save_model_steps 50000 \\
    --save_images_epochs 25000 \\
    --num_train_steps 1000 \\
    --num_inference_steps 1000 \\
    --train_scheduler ddpm \\
    --test_scheduler ddpm \\
    """
    # Execute the command
    os.system(command)

# Notify the user about the process
print("Commands executed for all variable combinations.")
