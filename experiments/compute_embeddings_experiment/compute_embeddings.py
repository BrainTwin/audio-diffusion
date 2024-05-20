import subprocess

def run_fadtk(model_name, dir1, dir2):
    command = ["fadtk", model_name, dir1, dir2]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        
        print(f"Command: {' '.join(command)}")
        print("Output:")
        print(result.stdout)
        if result.stderr:
            print("Error:")
            print(result.stderr)
    except Exception as e:
        print(f"An error occurred while running the command: {' '.join(command)}")
        print(f"Exception: {e}")

def main():
    model_names = [
        "clap-laion-audio", "clap-laion-music", "vggish", 
        "MERT-v1-95M", "clap-2023", "encodec-emb", "encodec-emb-48k"
    ]
    
    dir_pairs = [
        ("cache/drum_samples/waveform", "cache/fma_pop/waveform"),
        ("cache/musiccaps/waveform", "cache/spotify_sleep_dataset/waveform")
    ]
    
    for dir1, dir2 in dir_pairs:
        for model_name in model_names:
            print(f'Now calculating embeddings with model name {model_name}')
            print(f'...and datasets {dir1} and {dir2}')
            run_fadtk(model_name, dir1, dir2)

if __name__ == "__main__":
    main()
