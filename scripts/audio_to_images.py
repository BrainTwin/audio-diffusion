import argparse
import io
import logging
import os
import re

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as PILImage
import soundfile as sf
from datasets import Dataset, DatasetDict, Features, Image, Value, Array3D
from diffusers.pipelines.audio_diffusion import Mel
from tqdm.auto import tqdm

# must deprecate diffusers library to allow for audio-diffusion huggingface library!
# https://github.com/huggingface/diffusers/issues/6463

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger("audio_to_images")

# ==========================================================================================
# Mel spectrogram generation code is taken from the official BigVGAN repository
# https://github.com/NVIDIA/BigVGAN
import torch
from librosa.filters import mel as librosa_mel_fn

mel_basis_cache = {}
hann_window_cache = {}

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)

def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int = None,
    center: bool = False,
) -> torch.Tensor:
    """
    Calculate the mel spectrogram of an input signal.
    This function uses slaney norm for the librosa mel filterbank (using librosa.filters.mel) and uses Hann window for STFT (using torch.stft).

    Args:
        y (torch.Tensor): Input signal.
        n_fft (int): FFT size.
        num_mels (int): Number of mel bins.
        sampling_rate (int): Sampling rate of the input signal.
        hop_size (int): Hop size for STFT.
        win_size (int): Window size for STFT.
        fmin (int): Minimum frequency for mel filterbank.
        fmax (int): Maximum frequency for mel filterbank. If None, defaults to half the sampling rate (fmax = sr / 2.0) inside librosa_mel_fn
        center (bool): Whether to pad the input to center the frames. Default is False.

    Returns:
        torch.Tensor: Mel spectrogram.
    """
    if torch.min(y) < -1.0:
        print(f"[WARNING] Min value of input waveform signal is {torch.min(y)}")
    if torch.max(y) > 1.0:
        print(f"[WARNING] Max value of input waveform signal is {torch.max(y)}")

    device = y.device
    key = f"{n_fft}_{num_mels}_{sampling_rate}_{hop_size}_{win_size}_{fmin}_{fmax}_{device}"

    if key not in mel_basis_cache:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis_cache[key] = torch.from_numpy(mel).float().to(device)
        hann_window_cache[key] = torch.hann_window(win_size).to(device)

    mel_basis = mel_basis_cache[key]
    hann_window = hann_window_cache[key]

    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(
        y.unsqueeze(1), (padding, padding), mode="reflect"
    ).squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = spectral_normalize_torch(mel_spec)

    return mel_spec


def get_mel_spectrogram(wav, h):
    """
    Generate mel spectrogram from a waveform using given hyperparameters.

    Args:
        wav (torch.Tensor): Input waveform.
        h: Hyperparameters object with attributes n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax.

    Returns:
        torch.Tensor: Mel spectrogram.
    """
    return mel_spectrogram(
        wav,
        h["n_fft"],
        h["num_mels"],
        h["sampling_rate"],
        h["hop_size"],
        h["win_size"],
        h["fmin"],
        h["fmax"],
    )


    
def get_bigvgan_config(bigvgan_model_name):
    cfg = {
        "n_fft": None,
        "num_mels": None,
        "sampling_rate": None,
        "hop_size": None,
        "win_size": None,
        "fmin": None,
        "fmax": None,
    }
    
    if bigvgan_model_name == 'bigvgan_v2_24khz_100band_256x':
        cfg["n_fft"] = 1024
        cfg["num_mels"] = 100
        cfg["sampling_rate"] = 24000
        cfg["hop_size"] = 256
        cfg["win_size"] = 1024
        cfg["fmin"] = 0
        cfg["fmax"] = 12000
        
    elif bigvgan_model_name == 'bigvgan_v2_44khz_128band_256x':
        cfg["n_fft"] = 1024
        cfg["num_mels"] = 128
        cfg["sampling_rate"] = 44100
        cfg["hop_size"] = 256
        cfg["win_size"] = 1024
        cfg["fmin"] = 0
        cfg["fmax"] = 22050
        
    elif bigvgan_model_name == 'bigvgan_base_24khz_100band':
        cfg["n_fft"] = 1024
        cfg["num_mels"] = 100
        cfg["sampling_rate"] = 24000
        cfg["hop_size"] = 256
        cfg["win_size"] = 1024
        cfg["fmin"] = 0
        cfg["fmax"] = 12000
        
    elif bigvgan_model_name == 'bigvgan_24khz_100band':
        cfg["n_fft"] = 1024
        cfg["num_mels"] = 100
        cfg["sampling_rate"] = 24000
        cfg["hop_size"] = 256
        cfg["win_size"] = 1024
        cfg["fmin"] = 0
        cfg["fmax"] = 12000
 
 
    else:
        raise ValueError(f"Model name '{bigvgan_model_name}' has not been implemented yet.")
    
    return cfg



def split_mel(mel, target_x_res):
    """
    Chunk the mel-spectrogram into appropriate lenth ranges, according to target_x_res
    """
    _, freq_bins, time_steps = mel.shape
    chunks = []
    start_indices = list(range(0, time_steps - target_x_res + 1, target_x_res))

    # last overlapping chunk
    if start_indices[-1] + target_x_res < time_steps:
        start_indices.append(time_steps - target_x_res)

    # create chunks
    for start_idx in start_indices:
        end_idx = start_idx + target_x_res
        chunks.append(mel[:, :, start_idx:end_idx])

    return chunks


    
def mel_to_image(mel):
    fig, ax = plt.subplots(figsize=(mel.shape[1] / 100, mel.shape[0] / 100), dpi=100)
    ax.imshow(mel.squeeze(0).cpu().numpy(), origin='lower', aspect='auto', cmap='inferno')
    ax.axis('off')

    buf = io.BytesIO()
    try:
        plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.1)  # Increased pad_inches to ensure image fits
        buf.seek(0)

        image = PILImage.open(buf)
    except Exception as e:
        print(f"Error saving image: {e}")
        image = None

    plt.close(fig)

    return image
# ==========================================================================================
    

def process_audio(audio_file, sample_rate, num_channels):
    """
    Load, resample, and enforce the number of channels for an audio file.
    """
    y, sr = librosa.load(audio_file, sr=sample_rate, mono=False)

    # Check and enforce number of channels
    if len(y.shape) == 1:  # Mono file
        if num_channels == 2:
            raise ValueError(f"Audio file {audio_file} is mono but 2 channels were specified.")
        elif num_channels == 1:
            return librosa.to_mono(y), sample_rate
    elif len(y.shape) == 2:  # Stereo file
        if num_channels == 1:
            return librosa.to_mono(y), sample_rate
        elif num_channels == 2:
            return y, sample_rate

    raise ValueError(f"Unexpected number of channels in {audio_file}: {y.shape}")

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    audio_files = [
        os.path.join(args.input_dir, file)
        for file in os.listdir(args.input_dir)
        if os.path.isfile(os.path.join(args.input_dir, file)) and re.search("\\.(mp3|wav|m4a)$", file, re.IGNORECASE)
    ]
    print(audio_files)
    
    create_dataset(
        args,
        audio_files,
        )
    
def create_dataset(args, audio_files):
    """
    Function to load audio files, convert them into mel-spectrograms, and then create an appropriate dataset.
    
    The method of dataset creation depends on args.mel_spec_method, which also determines the specifications of the dataset
    """
    examples = []
    
    # 'image' method as proposed in:
    # https://github.com/teticio/audio-diffusion
    # by Robert Dargavel Smith
    if args.mel_spec_method == "image":
        mel = Mel(
            x_res=args.resolution[0],
            y_res=args.resolution[1],
            hop_length=args.hop_length,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
        )
            
        try:
            for audio_file in tqdm(audio_files):
                try:
                    # Process the audio file
                    processed_audio, sr = process_audio(audio_file, args.sample_rate, args.num_channels)

                    # Save the processed audio to a temporary file
                    temp_audio_file = f"temp_{os.path.basename(audio_file)}"
                    sf.write(temp_audio_file, processed_audio.T, sr, format='WAV')

                    # Load the processed audio into the Mel class
                    mel.load_audio(temp_audio_file)
                    os.remove(temp_audio_file)

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(e)
                    continue

                for slice in range(mel.get_number_of_slices()):
                    image = mel.audio_slice_to_image(slice)
                    assert image.width == args.resolution[0] and image.height == args.resolution[1], "Wrong resolution"
                    # Skip completely silent slices
                    if all(np.frombuffer(image.tobytes(), dtype=np.uint8) == 255):
                        logger.warn("File %s slice %d is completely silent", audio_file, slice)
                        continue
                    with io.BytesIO() as output:
                        image.save(output, format="PNG")
                        bytes = output.getvalue()
                    examples.extend(
                        [
                            {
                                "image": {"bytes": bytes},
                                "audio_file": audio_file,
                                "slice": slice,
                            }
                        ]
                    )
        except Exception as e:
            print(e)
        finally:
            if len(examples) == 0:
                logger.warn("No valid audio files were found.")
                return
            ds = Dataset.from_pandas(
                pd.DataFrame(examples),
                features=Features(
                    {
                        "image": Image(),
                        "audio_file": Value(dtype="string"),
                        "slice": Value(dtype="int16"),
                    }
                ),
            )
            dsd = DatasetDict({"train": ds})
            dsd.save_to_disk(os.path.join(args.output_dir))
            if args.push_to_hub:
                dsd.push_to_hub(args.push_to_hub)
                
                
    # The BigVGAN method proposed by NVIDIA
    # using code taken from:
    # https://github.com/NVIDIA/BigVGAN/blob/main/meldataset.py
    elif args.mel_spec_method == "bigvgan":
        
        def data_generator(audio_files, bigvgan_config, resolution, num_channels):
            device = 'cpu'
            for audio_file in tqdm(audio_files):
                try:
                    # Process the audio file
                    wav, sr = librosa.load(audio_file, sr=bigvgan_config["sampling_rate"], mono=True)
                    wav = torch.FloatTensor(wav).unsqueeze(0)
                    mel = get_mel_spectrogram(wav, bigvgan_config).to(device)

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
                    continue

                # Split mel spectrogram and yield each chunk
                for i, chunk in enumerate(split_mel(mel, resolution[0])):
                    assert chunk.ndimension() == 3, f"Expected a tensor with 3 dimensions, but got {chunk.ndimension()} dimensions."
                    assert chunk.shape[2] == resolution[0] and chunk.shape[1] == resolution[1], "Wrong resolution"

                    # Create the image from the chunk (if necessary)
                    mel_image = mel_to_image(chunk)
                    img_byte_arr = io.BytesIO()
                    mel_image.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()

                    # Convert the tensor to a numpy array with float32 type
                    mel_numpy = chunk.cpu().numpy().astype("float32")
                    
                    cutoff = 2.1
                    chunk = torch.clamp(chunk, min=None, max=cutoff)  # Cap values at the cutoff
                    
                    # Yield the example as a dictionary
                    yield {
                        "mel": chunk,
                        "image": {"bytes": img_byte_arr},
                        "audio_file": audio_file,
                        "slice": i,
                    }

        def create_dataset(audio_files, bigvgan_config, resolution, num_channels):
            # Define the features for the dataset
            features = Features(
                {
                    "mel": Array3D(dtype="float32", shape=(num_channels, resolution[1], resolution[0])),  # Shape of mel tensor
                    "audio_file": Value(dtype="string"),
                    "slice": Value(dtype="int16"),
                }
            )

            # Create the dataset using the generator
            ds = Dataset.from_generator(
                data_generator,
                gen_kwargs={"audio_files": audio_files, "bigvgan_config": bigvgan_config, "resolution": resolution, "num_channels": num_channels},
                features=features
            )
            
            return ds
        assert args.bigvgan_model is not None

        bigvgan_config = get_bigvgan_config(args.bigvgan_model)
        

        # Create the dataset
        ds = create_dataset(audio_files, bigvgan_config, args.resolution, args.num_channels)

        # Create the DatasetDict and save
        dsd = DatasetDict({"train": ds})
        dsd.save_to_disk(os.path.join(args.output_dir))

        # Optionally push to hub
        if args.push_to_hub:
            dsd.push_to_hub(args.push_to_hub)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset of Mel spectrograms from directory of audio files.")
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--resolution", type=str, default="256", help="Either square resolution or width,height.")
    parser.add_argument("--max_examples", type=int, default=None, help="If set, will limit the amount of wav files to include in the dataset.")
    parser.add_argument("--mel_spec_method", type=str, choices=["image", "bigvgan"])
    parser.add_argument("--bigvgan_model", type=str, choices=[
                            "bigvgan_v2_44khz_128band_512x",
                            "bigvgan_v2_44khz_128band_256x",
                            "bigvgan_v2_24khz_100band_256x",
                            "bigvgan_v2_22khz_80band_256x",
                            "bigvgan_v2_22khz_80band_fmax8k_256x",
                            "bigvgan_24khz_100band",
                            "bigvgan_base_24khz_100band",
                            "bigvgan_22khz_80band",
                            "bigvgan_base_22khz_80band"
                        ], 
                        help='If the bigvgan method for mel-spectrogram generation is selected, pick which pre-trained model to use as the configuration.')
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--push_to_hub", type=str, default=None)
    parser.add_argument("--sample_rate", type=int, default=22050, help="Target sample rate for audio files.")
    parser.add_argument("--num_channels", type=int, choices=[1, 2], default=1, help="Number of audio channels (1=mono, 2=stereo).")
    parser.add_argument("--n_fft", type=int, default=1024)
    args = parser.parse_args()

    if args.input_dir is None:
        raise ValueError("You must specify an input directory for the audio files.")

    # Handle the resolutions.
    try:
        args.resolution = (int(args.resolution), int(args.resolution))
    except ValueError:
        try:
            args.resolution = tuple(int(x) for x in args.resolution.split(","))
            if len(args.resolution) != 2:
                raise ValueError
        except ValueError:
            raise ValueError("Resolution must be a tuple of two integers or a single integer.")
    assert isinstance(args.resolution, tuple)

    main(args)
