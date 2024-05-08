import argparse
import os
import re
import subprocess
from torch.utils.tensorboard import SummaryWriter

def run_fad_calculation(model_name, reference_path, generated_path):
    """
    Runs the FAD calculation as a subprocess and captures the output.
    """
    try:
        # Build the command for subprocess
        command = ['fadtk', model_name, reference_path, generated_path]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        main_pattern = r'__main__.py:\d+'
        result = re.sub(main_pattern, '', result.stdout)
        space_pattern = r'[\n\t]+'
        result = re.sub(space_pattern, '', result)
        #
        pattern = r'\d+\.\d+(?=\s*$)'
        match = re.search(pattern, result, re.M) # use re.MULTILINE
        
        
        if match:
            fad_score = float(match.group())
            return fad_score
        else:
            raise ValueError("FAD score not found in subprocess output.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to run subprocess: {e}")

def log_to_tensorboard(log_dir, step, fad_score, metric_log_name):
    """
    Logs the given FAD score to TensorBoard.
    """
    with SummaryWriter(log_dir) as writer:
        writer.add_scalar(metric_log_name, fad_score, step)

def main(args):
    # Check if the metric to calculate is Frechet Audio Distance
    if args.metric == "frechet_audio_distance":
        print("Calculating Frechet Audio Distance...")
        for model_name in args.model_names:
            fad_score = run_fad_calculation(model_name, args.reference_path, args.generated_path)
            print(f"{model_name} - Frechet Audio Distance Score: {fad_score}")
            tensorboard_log_dir = os.path.join(args.log_dir, args.generated_path)
            metric_log_name = f'{args.metric}_{model_name}'
            log_to_tensorboard(tensorboard_log_dir, args.log_step, fad_score, metric_log_name)
    else:
        print("No supported metrics requested.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate audio samples using various metrics.")
    parser.add_argument("--reference_path", type=str, required=True, help="Path to the reference audio samples.")
    parser.add_argument("--generated_path", type=str, required=True, help="Path to the generated audio samples.")
    parser.add_argument("--metric", type=str, choices=['frechet_audio_distance'], help="Specify which metric to calculate.")
    parser.add_argument(
        "--model_names", 
        type=str, 
        nargs='+',
        # choices=[
        #     "clap-2023",
        #     "clap-laion-audio",
        #     "clap-laion-music",
        #     "encodec-emb",
        #     "MERT-v1-95M-layer",
        #     "vggish",
        #     "dac-44kHz",
        #     "cdpam-acoustic",
        #     "cdpam-content",
        #     "w2v2-base",
        #     "w2v2-large",
        #     "hubert-base",
        #     "hubert-large",
        #     "wavlm-base",
        #     "wavlm-base-plus",
        #     "wavlm-large",
        #     "whisper-tiny",
        #     "whisper-base",
        #     "whisper-small",
        #     "whisper-medium",
        #     "whisper-large"
        # ],
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