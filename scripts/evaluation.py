import argparse
import os
import re
import subprocess
from torch.utils.tensorboard import SummaryWriter

def run_fad_calculation(model_name, reference_path, generated_path):
    """
    Runs the FAD calculation as a subprocess and captures the output.
    If the subprocess fails, it returns None instead of raising an error.
    """
    try:
        command = ['fadtk', model_name, reference_path, generated_path]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        main_pattern = r'__main__.py:\d+'
        result = re.sub(main_pattern, '', result.stdout)
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
    with SummaryWriter(log_dir) as writer:
        writer.add_scalar(metric_log_name, fad_score, step)

def get_dataset_name_from_path(path):
    if 'drum_samples' in path: return 'drum_samples'
    elif 'spotify_sleep_dataset' in path: return 'spotify_sleep_dataset'
    elif 'musiccaps' in path: return 'musiccaps'
    elif 'fma_pop' in path: return 'fma_pop'
    else: return None

def main(args):
    if "frechet_audio_distance" in args.metric:
        print("Calculating Frechet Audio Distance...")
        for reference_path in args.reference_paths:
            for model_name in args.model_names:
                fad_score = run_fad_calculation(model_name, reference_path, args.generated_path)
                if fad_score is not None:
                    print(f"{model_name} - Frechet Audio Distance Score: {fad_score} (Reference: {reference_path})")
                    reference_dataset_name = get_dataset_name_from_path(reference_path)
                    generated_dataset_name = get_dataset_name_from_path(args.generated_path)
                    metric_log_name = f'fad_{model_name}_ref_{reference_dataset_name}_gen_{generated_dataset_name}'
                    log_to_tensorboard(args.log_dir, args.log_step, fad_score, metric_log_name)
                else:
                    print(f"Error: Could not calculate FAD score for model {model_name} with reference {reference_path}. Skipping to next model.")
    else:
        print("No supported metrics requested.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate audio samples using various metrics.")
    parser.add_argument("--reference_paths", type=str, nargs='+', required=True, help="Paths to the reference audio samples.")
    parser.add_argument("--generated_path", type=str, required=True, help="Path to the generated audio samples.")
    parser.add_argument("--metric", type=str, nargs='+', default=['frechet_audio_distance'], choices=['frechet_audio_distance'], help="Specify which metric to calculate.")
    parser.add_argument(
        "--model_names", 
        type=str, 
        nargs='+',
        choices=[
            'clap-2023', 
            'clap-laion-audio', 
            'clap-laion-music', 
            'vggish', 
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
