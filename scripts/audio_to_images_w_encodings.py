import argparse
import ast
import io
import logging
import numpy as np
import os
import re
import pandas as pd
from tqdm.auto import tqdm
import torch
from transformers import T5Tokenizer, T5EncoderModel
from diffusers.pipelines.audio_diffusion import Mel

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger("audio_to_images_w_encodings")

def load_encodings(csv_path, idxs):
    df = pd.read_csv(csv_path)  # Ensure 'filename' is the index without the extension
    df = df.loc[idxs]
    df = df[df['Genres'] != '[unknown]']
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5EncoderModel.from_pretrained('t5-small')
    
    encodings = {}
    for idx, row in tqdm(df.iterrows(), desc='Generating encodings!'):
        # Parse the 'Genres' column string into a list
        try:
            genres_list = ast.literal_eval(row['Genres'])
        except ValueError:
            logger.warn(f"Failed to parse genres for file {idx}")
            continue
        
        text = ' '.join(genres_list)  # Now 'genres_list' is an actual list
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        encodings[idx] = outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # No need to add '.wav'
    return encodings

def main(args):
    mel = Mel(
        x_res=args.resolution[0],
        y_res=args.resolution[1],
        hop_length=args.hop_length,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    audio_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(args.input_dir)
        for file in files if re.search(r"\.(mp3|wav|m4a)$", file, re.IGNORECASE)
    ]
    
    idxs = [int(os.path.splitext(os.path.basename(audio_file))[0]) for audio_file in audio_files]
    encodings = load_encodings(args.dataframe_path, idxs)

    examples = []
    for audio_file in tqdm(audio_files):
        mel.load_audio(audio_file)
        
        file_name = int(os.path.splitext(os.path.basename(audio_file))[0])  # Extract base filename without extension
        if file_name not in encodings:
            logger.warn(f"Encoding for file {file_name} not found")
            continue
        encoding = encodings[file_name]

        for slice in range(mel.get_number_of_slices()):
            image = mel.audio_slice_to_image(slice)
            if all(np.frombuffer(image.tobytes(), dtype=np.uint8) == 255):
                logger.warn(f"File {audio_file} slice {slice} is completely silent")
                continue
            with io.BytesIO() as output:
                image.save(output, format="PNG")
                bytes = output.getvalue()
            examples.append({
                "image": {"bytes": bytes},
                "audio_file": audio_file,
                "slice": slice,
                "encoding": encoding,
            })

    if examples:
        from datasets import Dataset, DatasetDict, Features, Image, Value, Sequence
        ds = Dataset.from_pandas(pd.DataFrame(examples))
        dsd = DatasetDict({"train": ds})
        dsd.save_to_disk(args.output_dir)
        if args.push_to_hub:
            dsd.push_to_hub(args.push_to_hub)
    else:
        logger.warn("No valid audio files were found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset of Mel spectrograms from directory of audio files.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--resolution", type=str, default="256", help="Either square resolution or width,height.")
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--push_to_hub", type=str)
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--dataframe_path", type=str, required=True, help="Path to the CSV file with metadata")
    args = parser.parse_args()

    if ',' in args.resolution:
        args.resolution = tuple(map(int, args.resolution.split(',')))
    else:
        args.resolution = (int(args.resolution), int(args.resolution))

    main(args)
