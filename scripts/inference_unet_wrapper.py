import subprocess
from pathlib import Path
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
import threading
import time
from pydub import AudioSegment
import os

PRETRAINED_MODEL_PATH = "../audio_diffusion_models/model_step_90000"#"/home/kidrm2/workspace/audio_diffusion/audio_diffusion_models/model_step_90000"
MEL_SPEC_METHOD = "image"
NUM_IMAGES = "1"
NUM_INFERENCE_STEPS = "100"
N_ITER = "32"
SEEDS = [# Runs the main inference script to generate 1 sample with the specified seed values respectively.
    #0, 1, 2, 3, 4, 5, 6, 9#, 7, 8
    0
] 
EVAL_BATCH_SIZE = "32"
SCHEDULER = "ddpm"

NORMALIZATION_TARGET_dBFS = -20.0
FADE_IN_DURATION = 600
FADE_OUT_DURATION = 600
PROCESSED_AUDIO_PATH = Path(PRETRAINED_MODEL_PATH) / "samples" / "processed_audio"

def match_target_amplitude(audio, target_dBFS):
    dBFS_difference = target_dBFS - audio.dBFS
    return audio.apply_gain(dBFS_difference)

def add_fade_in_out(audio, fade_in_duration, fade_out_duration): # Fade in and out duration in ms
    faded_audio = audio.fade_in(fade_in_duration).fade_out(fade_out_duration)
    #faded_audio.export(output_path, format="wav")
    return faded_audio

class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:  # Only process file creation events
            print(f"New file created: {event.src_path}")
            src_path = Path(event.src_path)
            if src_path.suffix == '.wav' and not str(src_path).endswith("processed.wav"):
                print(f"{str(src_path)} is a wav file. Normalizing and adding fade.")
                audio = AudioSegment.from_file(str(src_path), format=src_path.suffix[1:])
                processed_audio = match_target_amplitude(audio, NORMALIZATION_TARGET_dBFS)
                
                processed_audio = add_fade_in_out(processed_audio, fade_in_duration=FADE_IN_DURATION, fade_out_duration=FADE_OUT_DURATION)
                processed_audio.export(
                    str(PROCESSED_AUDIO_PATH / src_path.stem) + "_processed" + src_path.suffix,
                    format=src_path.suffix[1:]
                )


def start_diffusion_output_watchdog(diffusion_output_dir):
    event_handler = NewFileHandler()
    observer = Observer()
    observer.schedule(event_handler, diffusion_output_dir, recursive=True) # Set recursive=True to monitor subdirectories
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

def main():
    os.makedirs(str(PROCESSED_AUDIO_PATH), exist_ok=True)
    diffusion_output_watchdog_thread = threading.Thread(target=start_diffusion_output_watchdog, args=(PRETRAINED_MODEL_PATH,), daemon=True)
    diffusion_output_watchdog_thread.start()

    for seed in SEEDS:
        command = ["python", "scripts/inference_unet.py",
            "--pretrained_model_path", PRETRAINED_MODEL_PATH,
            "--mel_spec_method", MEL_SPEC_METHOD,
            "--num_images", NUM_IMAGES,
            "--num_inference_steps", NUM_INFERENCE_STEPS,
            "--n_iter", N_ITER,
            "--seed", str(seed),
            "--eval_batch_size", EVAL_BATCH_SIZE,
            "--scheduler", SCHEDULER
        ]
        try:
            #result = subprocess.run(command, capture_output=True, text=True)
            #result = subprocess.run(' '.join(command), shell=True)
            result = subprocess.run(command)
            
            print(f"Command: {' '.join(command)}")
            print("Output:")
            print(result.stdout) # Will be None if capture_outputs is not set
            if result.stderr:
                print("Error:")
                print(result.stderr)
        except Exception as e:
            print(f"An error occurred while running the command: {' '.join(command)}")
            print(f"Exception: {e}")

    #diffusion_output_watchdog_thread.join()
    # TODO: change this from sleep to diffusion_output_watchdog_thread.join() without daemon=True in watchdog thread
    # Need the while loop in start_diffusion_output_watchdog() to listen for an event i.e. all calls in the main() for loop
    # to the inference script to be completed.
    # join is required to ensure the last track is processed before main exits and the watchdog thread is terminated.
    # For now just wait for a few seconds in main befor exiting.
    time.sleep(10)
    print("main() exiting.")

if __name__ == "__main__":
    main()