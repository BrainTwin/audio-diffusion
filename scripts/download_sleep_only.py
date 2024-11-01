import os
import requests
import pandas as pd
from tqdm import tqdm

save_dir = '/home/th716/audio-diffusion/cache/spotify_sleep_dataset/waveform_sleep_only'
os.makedirs(save_dir, exist_ok=True) 

def download_audio(sample_url, track_id):
    try:
        response = requests.get(sample_url, stream=True)
        if response.status_code == 200:
            file_path = os.path.join(save_dir, f"{track_id}.wav")
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {track_id}")
        else:
            print(f"Failed to download {track_id}: HTTP {response.status_code}")
    except Exception as e:
        print(f"Error downloading {track_id}: {e}")

def main():
    df = pd.read_csv('/home/th716/audio-diffusion/spotify_sleep_dataset/sleep_only_dataset.csv')
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading audio files"):
        sample_url = row['SampleURL']
        track_id = row['TrackID']
        download_audio(sample_url, track_id)

if __name__ == "__main__":
    main()
