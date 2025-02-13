import os
import random
import torch
import numpy as np
import argparse
from tqdm.auto import tqdm
from datasets import load_dataset, load_from_disk
from librosa.feature import inverse as librosa_inverse
from diffusers.pipelines.audio_diffusion import Mel
import soundfile as sf
from torchvision.transforms import Compose, Normalize, ToTensor


def convert_mel_to_audio(args):
    if args.generation_method not in ["image", "bigvgan"]:
        raise ValueError("Invalid generation method. Use 'image' or 'bigvgan'.")



    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_from_disk(args.mel_spec_path)
    indices = list(range(len(dataset["train"])))  # Create an index list
    random.shuffle(indices)  # Shuffle the indices
    subset_indices = indices[:args.num_samples]  # Take a subset
    
    resolution = dataset["train"][0]["image"].size
    
    if args.generation_method == "image":
        mel = Mel(
            x_res=resolution[0],
            y_res=resolution[1],
            hop_length=args.hop_length,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            n_iter=args.griffin_lim_iters
        )

    print(f"Processing {len(dataset)} mel-spectrogram files...")

    for i in tqdm(subset_indices, desc="Converting mel to audio"):
        mel_data = dataset["train"][i]
        mel_path = os.path.join(args.output_dir, mel_data['audio_file'].split('/')[-1])

        # Convert mel to audio
        if args.generation_method == "image":
            if args.griffin_lim_iters is None:
                raise ValueError("Griffin-Lim iterations must be provided for 'image' method.")
            audio = mel.image_to_audio(mel_data["image"])
            
        sf.write(mel_path, audio, samplerate=args.sample_rate)

    print(f"Audio files saved to {args.output_dir}")

# Main function with argparse for argument parsing
def main():
    parser = argparse.ArgumentParser(description="Convert mel-spectrograms to audio.")
    parser.add_argument('--mel_spec_path', type=str, help="Path to the mel-spectrogram dataset.")
    parser.add_argument('--generation_method', type=str, choices=["image", "bigvgan"], help="Method to generate audio: 'image' for Griffin-Lim or 'bigvgan'.")
    parser.add_argument('--griffin_lim_iters', type=int, default=None, help="Number of Griffin-Lim iterations for 'image' method.")
    parser.add_argument('--bigvgan_model_name', type=str, default=None, help="BigVGAN model name for 'bigvgan' method.")
    parser.add_argument('--output_dir', type=str, default="output_audio", help="Directory to save the output audio files.")
    parser.add_argument('--num_samples', type=int, default=None, help="Number of mel-spectrogram files to process (optional).")
    parser.add_argument('--sample_rate', type=int, default=None, help="Sample rate of the mel-spectrograms/audio to generate (must match!).")
    parser.add_argument('--hop_length', type=int, default=256, help="Hop length that the mel-specs were generated with.")
    parser.add_argument('--n_fft', type=int, default=1024, help="number of fast fourier trasnforms that the mel-spec dataset was generated with")
    parser.add_argument('--top_db', type=int, default=None, help="loudest frequency (in decibels)")
    
    args = parser.parse_args()

    convert_mel_to_audio(args)

if __name__ == "__main__":
    main()
