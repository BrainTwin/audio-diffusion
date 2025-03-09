import argparse
import os
import re
import requests
from statistics import mean
import subprocess
import sys

print(sys.path)

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from torchaudio.transforms import MelSpectrogram
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/pann')))

# Now you can import the model
from models import Cnn14  # Import the Cnn14 model from models.py 

# Function to load the PANN model from the local .pth file
def load_pann_model(checkpoint_path):
    model = Cnn14(
        sample_rate=16000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=527
    )
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    model.fc = torch.nn.Identity()  # Replace the final classification layer with identity to get penultimate layer features
    model.eval()
    return model

# Function to extract features using the PANN model
def extract_pann_features(model, audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:  # Resample if necessary
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    mel_spectrogram = MelSpectrogram()(waveform)
    mel_spectrogram = Normalize(mean=[-13.8], std=[17.2])(mel_spectrogram)
    
    with torch.no_grad():
        features = model(mel_spectrogram.unsqueeze(0)).squeeze()
    
    return features

# Function to calculate KL Divergence
def kl_divergence(p, q):
    return torch.sum(p * torch.log(p / q))

def calculate_kl_divergence(reference_path, generated_path, pann_model):
    # Get the list of all files with the required extensions in reference_path
    ref_paths = []
    for root, _, files in os.walk(reference_path):
        ref_paths.extend([os.path.join(root, file) for file in files if file.endswith(('.wav', '.mp3'))])

    # Get the list of all files with the required extensions in generated_path
    gen_paths = []
    for root, _, files in os.walk(generated_path):
        gen_paths.extend([os.path.join(root, file) for file in files if file.endswith(('.wav', '.mp3'))])
    
    # Sort paths to ensure they correspond to each other
    ref_paths.sort()
    gen_paths.sort()

    kl_scores = []
    for ref_path, gen_path in zip(ref_paths, gen_paths):
        reference_features = extract_pann_features(pann_model, ref_path)
        generated_features = extract_pann_features(pann_model, gen_path)
        
        reference_features = F.softmax(reference_features, dim=0)
        generated_features = F.softmax(generated_features, dim=0)
        
        kl_score = kl_divergence(reference_features, generated_features)
        kl_scores.append(kl_score.item())
    
    # Return the average KL divergence score
    return mean(kl_scores)


def run_fad_calculation(model_name, reference_path, generated_path):
    """
    Runs the FAD calculation as a subprocess and captures the output.
    If the subprocess fails, it returns None instead of raising an error.
    """
    try:
        command = ['fadtk', model_name, reference_path, generated_path]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        main_pattern = r'__main__.py:\d+'
        result = re.sub(main_pattern, '', result.stderr)
        print(f'Raw result is: {result}\n\n')
        
        space_pattern = r'[\n\t]+'
        result = re.sub(space_pattern, '', result)

        pattern = r'\d+\.\d+(?=\s*$)'
        match = re.search(pattern, result, re.M)
        
        if match:
            fad_score = float(match.group())
            return fad_score
        else:
            print(f"Warning: FAD score not found in subprocess output for model {model_name}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to run subprocess for model {model_name}: {e}")
        return None

def log_to_tensorboard(log_dir, step, fad_score, metric_log_name):
    """
    Logs the given FAD score to TensorBoard.
    """
    os.makedirs(log_dir, exist_ok=True)
    with SummaryWriter(log_dir) as writer:
        writer.add_scalar(metric_log_name, fad_score, step)

def get_dataset_name_from_path(path):
    if 'drum_samples' in path: return 'drum_samples'
    elif 'spotify_sleep_dataset' in path: return 'spotify_sleep_dataset'
    elif 'musiccaps' in path: return 'musiccaps'
    elif 'fma_pop' in path: return 'fma_pop'
    else: return None


def main(args):
    # model_url = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
    # checkpoint_path = "../models/Cnn14_mAP=0.431.pth"
    
    # if not os.path.exists(checkpoint_path):
    #     print(f"Downloading PANN model from {model_url}...")
    #     download_pann_model(model_url, checkpoint_path)
    
    # pann_model = load_pann_model(checkpoint_path)  # Load PANN model with penultimate layer features
    
    for reference_path in args.reference_paths:
        print('======================================')
        print(f'\n\nNow running evaluation experiments for reference path: {reference_path}\n')
        if "frechet_audio_distance" in args.metric:
            print("Calculating Frechet Audio Distance...")
            for model_name in args.model_names:
                print('------------------------------------')
                print(f'\nNow running evaluation experiments for model: {model_name}\n')
                fad_score = run_fad_calculation(model_name, reference_path, args.generated_path)
                if fad_score is not None:
                    print(f"{model_name} - Frechet Audio Distance Score: {fad_score} (Reference: {reference_path})")
                    reference_dataset_name = get_dataset_name_from_path(reference_path)
                    generated_dataset_name = get_dataset_name_from_path(args.generated_path)
                    metric_log_name = f'fad_{model_name}_ref_{reference_dataset_name}_gen_{generated_dataset_name}'
                    log_to_tensorboard(args.log_dir, args.log_step, fad_score, metric_log_name)
                else:
                    print(f"Error: Could not calculate FAD score for model {model_name} with reference {reference_path}. Skipping to next model.")
            
            
        elif "kl_divergence" in args.metric:
            print("Calculating KL Divergence...")
            
            kl_score = calculate_kl_divergence(reference_path, args.generated_path, pann_model)
            print(f"{model_name} - KL Divergence Score: {kl_score} (Reference: {reference_path})")
            
            reference_dataset_name = get_dataset_name_from_path(reference_path)
            generated_dataset_name = get_dataset_name_from_path(args.generated_path)
            metric_log_name = f'kl_divergence_{model_name}_ref_{reference_dataset_name}_gen_{generated_dataset_name}'
            log_to_tensorboard(args.log_dir, args.log_step, kl_score, metric_log_name)
        
    else:
        print("No supported metrics requested.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate audio samples using various metrics.")
    parser.add_argument("--reference_paths", type=str, nargs='+', required=True, help="Paths to the reference audio samples.")
    parser.add_argument("--generated_path", type=str, required=True, help="Path to the generated audio samples.")
    parser.add_argument("--metric", type=str, nargs='+', default=['frechet_audio_distance'], choices=['frechet_audio_distance', 'kl_divergence'], help="Specify which metric to calculate.")
    parser.add_argument(
        "--model_names", 
        type=str, 
        nargs='+',
        choices=[
            'clap-2023', 
            'clap-laion-audio', # *
            'clap-laion-music', # * 
            'vggish', # *
            'MERT-v1-95M-1', 
            'MERT-v1-95M-2', 
            'MERT-v1-95M-3', 
            'MERT-v1-95M-4', 
            'MERT-v1-95M-5', 
            'MERT-v1-95M-6', 
            'MERT-v1-95M-7', 
            'MERT-v1-95M-8', 
            'MERT-v1-95M-9', 
            'MERT-v1-95M-10', 
            'MERT-v1-95M-11', 
            'MERT-v1-95M', 
            'encodec-emb', 
            'encodec-emb-48k'
        ],
        required=True, help="Model name for FAD calculation.")
    parser.add_argument("--log_dir", type=str, default='./models/evaluation', help="Directory for TensorBoard logs.")
    parser.add_argument("--log_step", type=int, default=0, help="Step index for logging.")

    args = parser.parse_args()
    main(args)
