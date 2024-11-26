import argparse
import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

def lengthen_audio(audio, sr, target_length):
    target_length_samples = target_length * sr
    audio_length_samples = len(audio)

    repeats_needed = (target_length_samples // audio_length_samples) + 1
    extended_audio = audio.copy()
    for _ in range(repeats_needed - 1):
        extended_audio = np.append(extended_audio, audio)

    final_audio = extended_audio[:target_length_samples]
    return final_audio

def process_audios(input_dir, output_dir, target_length):
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir), desc=f'Processing audio files in {input_dir}'):
        if filename.endswith('.wav') or filename.endswith('.mp3'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            audio, sr = librosa.load(input_path, sr=None)
            lengthened_audio = lengthen_audio(audio, sr, target_length)
            sf.write(output_path, lengthened_audio, sr)

            print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lengthen audio files by concatenating them.")
    parser.add_argument("--input_dir", type=str, help="Directory containing .wav or .mp3 files.")
    parser.add_argument("--output_dir", type=str, help="Directory to save the lengthened audio files.")
    parser.add_argument("--target_length", type=int, help="Target length of the audio in seconds.")

    args = parser.parse_args()
    process_audios(args.input_dir, args.output_dir, args.target_length)
