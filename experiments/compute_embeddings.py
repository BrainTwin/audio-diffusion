import subprocess

def run_fadtk(model_name, dir1, dir2):
    command = ["fadtk", model_name, dir1, dir2]
    result = subprocess.run(command, capture_output=True, text=True)
    
    print(f"Command: {' '.join(command)}")
    print("Output:")
    print(result.stdout)
    if result.stderr:
        print("Error:")
        print(result.stderr)

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
            run_fadtk(model_name, dir1, dir2)

if __name__ == "__main__":
    main()
