import argparse
import os
import re
import subprocess
import pandas as pd

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

def get_dataset_name_from_path(path):
    if 'drum_samples' in path: return 'drum_samples'
    elif 'spotify_sleep_dataset' in path: return 'spotify_sleep_dataset'
    elif 'musiccaps' in path: return 'musiccaps'
    elif 'fma_pop' in path: return 'fma_pop'
    else: return None

def main(args):
    results = []
    if "frechet_audio_distance" in args.metric:
        print("Calculating Frechet Audio Distance...")
        for reference_path in args.reference_paths:
            print('======================================')
            print(f'\n\nNow running evaluation experiments for reference path: {reference_path}\n')
            for generated_path in args.generated_paths:
                print(f'\nNow running evaluation experiments for generated path: {generated_path}\n')
                for model_name in args.model_names:
                    print('------------------------------------')
                    print(f'\nNow running evaluation experiments for model: {model_name}\n')
                    fad_score = run_fad_calculation(model_name, reference_path, generated_path)
                    if fad_score is not None:
                        print(f"{model_name} - Frechet Audio Distance Score: {fad_score} (Reference: {reference_path}, Generated: {generated_path})")
                        reference_dataset_name = get_dataset_name_from_path(reference_path)
                        generated_dataset_name = get_dataset_name_from_path(generated_path)
                        results.append({
                            'model_name': model_name,
                            'reference_dataset': reference_dataset_name,
                            'generated_dataset': generated_dataset_name,
                            'fad_score': fad_score
                        })
                    else:
                        print(f"Error: Could not calculate FAD score for model {model_name} with reference {reference_path}. Skipping to next model.")
    else:
        print("No supported metrics requested.")

    # Convert results to DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv('how_many_samples_for_fad.csv', index=False)
    print("Results saved to 'how_many_samples_for_fad.csv'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate audio samples using various metrics.")
    parser.add_argument("--reference_paths", type=str, nargs='+', required=True, help="Paths to the reference audio samples.")
    parser.add_argument("--generated_paths", type=str, nargs='+', required=True, help="Paths to the generated audio samples.")
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
    args = parser.parse_args()
    main(args)
