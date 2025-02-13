import argparse
import numpy as np
import os
from pathlib import Path
from PIL import Image
import random
import time
from tqdm import tqdm
import pickle

import librosa
import matplotlib.pyplot as plt
import pyloudnorm as pyln
import soundfile as sf
import torch
import torchaudio
from torchvision.utils import save_image
import torchvision.transforms as transforms

import sys
sys.path.insert(0, "/home/th716/audio-diffusion/submodules/diffusers/src/")

import diffusers
print('printing diffusers file first time')
print(diffusers.__file__)
from diffusers import AudioDiffusionPipeline
from diffusers import (AutoencoderKL, DDIMScheduler, DDPMScheduler,
                       UNet2DConditionModel, UNet2DModel, Mel)
from accelerate import Accelerator

# TODO
# - check mel spectrogram incorporation and if it can be set/changed. Esp with griffin lim iterations
# - integrate continuous inpainting and tested on debugger
# - 

def main(args):
    accelerator = Accelerator()

    model_path = args.pretrained_model_path
    output_path = os.path.join(model_path, 'samples')
    sch_tag = f'sch_{args.scheduler}_nisteps_{args.num_inference_steps}'
    images_path = os.path.join(output_path, 'images', sch_tag)
    audios_path = os.path.join(output_path, 'audio', sch_tag)
    
    if args.mel_spec_method == "image":
        gl_tag = f'gl{args.n_iter}'
        images_path = os.path.join(images_path, gl_tag)
        audios_path = os.path.join(output_path, gl_tag)
    elif args.mel_spec_method == "bigvgan":
        bgvg_tag = 'bigvgan'
        images_path = os.path.join(images_path, bgvg_tag)
        audios_path = os.path.join(output_path, bgvg_tag)
        
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(audios_path, exist_ok=True)

    generator = torch.Generator(device='cuda').manual_seed(args.seed)
    
    pipeline = AudioDiffusionPipeline.from_pretrained(model_path)
    pipeline.mel.config["n_iter"] = args.n_iter

    model = pipeline.unet
    vqvae = accelerator.prepare(pipeline.vqvae) if hasattr(pipeline, "vqvae") else None
    
    # Load encodings if provided
    encodings = None
    if args.encodings is not None:
        with open(args.encodings, "rb") as f:
            encodings = pickle.load(f)
        if isinstance(encodings, dict):
            encodings = {k: torch.tensor(v).to(accelerator.device) for k, v in encodings.items()}
        else:
            # Handle the case where encodings is not a dictionary
            encodings = torch.tensor(encodings).to(accelerator.device)
     
    # Prepare the model for generation
    model = pipeline.unet
    model = accelerator.prepare(model)
    model.eval()
    
    if args.scheduler == 'ddim':
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.scheduler.eta = args.eta
    
    if args.continuous_outpainting:
        # calculate how many outpaints we need to generate target length
        length_of_sample = model.config.sample_size[1] * (pipeline.mel.hop_length / pipeline.mel.sample_rate)
        num_iterations = int(np.ceil((args.target_length-length_of_sample)/((1-args.percentage_overlap)*length_of_sample)))

        random_wav_files = get_random_wav_files(args.gold_label_path, args.num_images)        
        for i, dataset_audio_path in enumerate(random_wav_files):
            image_path = os.path.join(images_path, f'image_{i}.png')
            audio_path = os.path.join(audios_path, f'audio_{i}.wav')
            

            mel_spec, audio = generate_continued_audio(
                dataset_audio_path,
                pipeline,
                model,
                target_length=args.target_length,
                target_loudness=-14,
                percentage=args.percentage_overlap,
                num_inference_steps=args.num_inference_steps,
                eta=args.eta,
                sample_rate=pipeline.mel.sample_rate
            )
            
            
            mel_spec.save(image_path)
            audio_tensor = torch.tensor(audio, dtype=torch.float32)
            audio_tensor = audio_tensor.unsqueeze(0)
            torchaudio.save(audio_path, audio_tensor, pipeline.mel.sample_rate)
        
    else:
        # Calculate number of batches
        num_batches = (args.num_images + args.eval_batch_size - 1) // args.eval_batch_size

        # Generate and save images and audio
        generated_count = 0
        for batch_idx in tqdm(range(num_batches), desc=f'Saving images and audio into folder: {output_path}'):
            batch_size = min(args.eval_batch_size, args.num_images - generated_count)
            
            with torch.no_grad():
                # If encodings are used, sample them according to the batch size
                encoding_sample = None
                if encodings is not None:
                    if isinstance(encodings, dict):
                        encoding_sample = torch.stack(list(encodings.values())[:batch_size])
                    else:
                        encoding_sample = encodings[:batch_size]
                
            
                images, (sample_rate, audios) = pipeline(
                    generator=generator, 
                    batch_size=batch_size,
                    return_dict=False, 
                    steps=args.num_inference_steps,
                    encoding=encoding_sample, 
                    eta=args.eta
                )

            for i in range(batch_size):
                image_index = generated_count + i + 1
                image_path = os.path.join(images_path, f'image_{image_index}.png')
                audio_path = os.path.join(audios_path, f'audio_{image_index}.wav')

                if args.mel_spec_method == "image":
                    images[i].save(image_path)

                    audio_tensor = torch.tensor(audios[i]).unsqueeze(0)  # Convert numpy array to tensor and add batch dimension
                    torchaudio.save(audio_path, audio_tensor, sample_rate)
                elif args.mel_spec_method == "bigvgan":
                    # TODO
                    # make the pipeline object return a tensor as well, not just a PIL image
                    # /home/th716/audio-diffusion/submodules/diffusers/src/diffusers/pipelines/audio_diffusion/pipeline_audio_diffusion.py
                    # for now we are working with images, and it works "fine" so we keep it at that
                    min_original, max_original = -11.5127, 2.1
                    min_pixel, max_pixel = 0, 255
                    def reverse_normalize(tensor):
                        return min_original + (tensor - min_pixel) * (max_original - min_original) / (max_pixel - min_pixel)

                    transform = transforms.Compose([transforms.PILToTensor()])
                    img_tensor = transform(images[i])
                    tensor_image = reverse_normalize(torch.tensor(img_tensor))

                    # Save the tensor to a file
                    tensor_path = os.path.join(images_path, f"tensor_{image_index}.pt")
                    torch.save(tensor_image, tensor_path)
                

            generated_count += batch_size

        print(f"All images and audios have been generated and saved to {output_path}.")
        
    
def get_random_wav_files(directory_path: str, n: int):
    
    """
    Given a directory path, return a list of n randomly selected .wav file paths.
    If there are not enough .wav files, raise a ValueError.

    :param directory_path: Path to the directory to search for .wav files
    :param n: Number of random .wav files to return
    :return: List of paths to .wav files
    :raises ValueError: If there are not enough .wav files in the directory
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"The provided path '{directory_path}' is not a valid directory.")
    wav_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.wav')]

    if len(wav_files) < n:
        raise ValueError(f"Not enough .wav files in the directory. Found {len(wav_files)}, but {n} required.")
    
    selected_files = random.sample(wav_files, n)

    return selected_files

    
def generate_continued_audio(
    audio_path,
    pipeline,
    model,
    target_length=30,
    target_loudness=-14.0,
    percentage=0.75,
    overlap_seconds=1.0,
    num_inference_steps=50,
    eta=0,
    sample_rate=22050
):
    """
    Generate continued audio by processing an initial audio input through a pipeline.

    Parameters:
        audio_path (str): Path to the input audio file.
        pipeline (object): Audio processing pipeline.
        model (object): Audio model with configuration.
        target_loudness (float): Target loudness level in LUFS.
        percentage (float): Percentage of sample length to use for generating new audio.
        num_iter (int): Number of iterations to generate new audio.
        overlap_seconds (float): Amount of overlap (in seconds) for seamless stitching.

    Returns:
        PIL.Image.Image: Image of the mel-spectrogram.
        np.ndarray: Numpy array of the continued audio waveform.
    """
    # Load and preprocess audio
    initial_audio, original_sample_rate = torchaudio.load(audio_path)
    initial_audio = torch.mean(initial_audio, dim=0, keepdim=True)  # Convert to mono

    # Resample audio
    sample_rate = pipeline.mel.get_sample_rate()
    initial_audio = librosa.resample(initial_audio.numpy()[0], orig_sr=original_sample_rate, target_sr=sample_rate)

    # Adjust loudness
    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(initial_audio)
    initial_audio = pyln.normalize.loudness(initial_audio, loudness, target_loudness)

    # Initialize parameters
    sample_length = model.config.sample_size[1] * (256 / sample_rate)
    duration_to_keep = sample_length * percentage
    overlap_samples = int(overlap_seconds * sample_rate)
    continued_audio = initial_audio[-int(duration_to_keep * sample_rate):]

    while len(continued_audio) < (target_length*sample_rate):
        # Process audio through the pipeline
        output = pipeline(
            batch_size=1,
            raw_audio=continued_audio[-int(duration_to_keep * sample_rate):],
            mask_start_secs=duration_to_keep,
            mask_end_secs=0,
            steps=args.num_inference_steps,
            eta=args.eta,
        )
        next_audio = output.audios[0, 0]

        # Normalize loudness
        continued_loudness = meter.integrated_loudness(continued_audio)
        next_loudness = meter.integrated_loudness(next_audio)
        next_audio = pyln.normalize.loudness(next_audio, next_loudness, continued_loudness)

        # Stitch audio with overlap
        # if len(continued_audio) > overlap_samples:
        #     transition = (
        #         continued_audio[-overlap_samples:] * np.linspace(1, 0, overlap_samples) +
        #         next_audio[:overlap_samples] * np.linspace(0, 1, overlap_samples)
        #     )
        #     continued_audio = np.concatenate([
        #         continued_audio[:-overlap_samples], transition, next_audio[overlap_samples:]
        #     ])
        # else:
        continued_audio = np.concatenate([continued_audio[:-int(duration_to_keep*sample_rate)], next_audio])
        
        

    # Generate the mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=continued_audio, sr=sample_rate, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Convert the mel-spectrogram to an image
    mel_height, mel_width = mel_spec_db.shape
    mel_image = (255 * (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())).astype(np.uint8)
    mel_image = Image.fromarray(mel_image).convert("L")

    return mel_image, continued_audio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using a trained model.")
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--mel_spec_method", type=str, default="image", choices=["image", "bigvgan"])
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--n_iter", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--scheduler", type=str, default='ddim', choices=['ddim', 'ddpm'])
    parser.add_argument("--eta", type=float, default=60, help="Set the eta parameter (works only if ddim scheduler is used). 0 = DDIM, 1.0 = DDPM.")
    parser.add_argument("--vae", type=str, default=None, help="Path to a pretrained VAE model.")
    parser.add_argument("--encodings", type=str, default=None, help="Path to pickle file containing encodings.")
    
    # arguments for continuous outpainting
    parser.add_argument("--continuous_outpainting", action='store_true', help="If set, continuous outpainting will be performed.")
    parser.add_argument("--target_length", type=float, default=60, help="Set the target length of the sample, in seconds.")
    parser.add_argument("--percentage_overlap", type=float, default=0.5, help="Specify the portion of the audio the model will see (keep) when outpainting.")
    parser.add_argument("--gold_label_path", type=str, default="/home/th716/rds/hpc-work/audio-diffusion/cache/spotify_sleep_dataset/waveform_sleep_only", help="The directory with the ground truth audios to start outpainting from.")

    args = parser.parse_args()
    main(args)