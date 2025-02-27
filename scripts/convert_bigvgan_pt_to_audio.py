import os
import torch
import argparse
import soundfile as sf
from tqdm import tqdm

def convert_bigvgan_tensor_to_audio(mel_tensor, output_path, bigvgan_model_name):
    device = 'cuda'
    import bigvgan
    
    model = bigvgan.BigVGAN.from_pretrained(f'nvidia/{bigvgan_model_name}', use_cuda_kernel=False)
    model.remove_weight_norm()
    model = model.eval().to(device)
    
    mel_tensor = mel_tensor.to(device)
    
    with torch.inference_mode():
        wav_gen = model(mel_tensor)  # [B, 1, T_time]
    wav_gen_float = wav_gen.squeeze(0).cpu()  # [1, T_time]
    
    sample_rate = model.h.sample_rate
    sf.write(output_path, wav_gen_float, sample_rate)

def process_directory(directory, bigvgan_model_name):
    pt_files = [f for f in os.listdir(directory) if f.endswith('.pt')]
    
    print(f"Found {len(pt_files)} .pt files in the directory.")
    
    for pt_file in tqdm(pt_files, desc="Processing files", unit="file"):
        pt_path = os.path.join(directory, pt_file)
        mel_tensor = torch.load(pt_path)
        
        wav_filename = os.path.splitext(pt_file)[0] + ".wav"
        wav_path = os.path.join(directory, wav_filename)
        
        convert_bigvgan_tensor_to_audio(mel_tensor, wav_path, bigvgan_model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .pt files to .wav using BigVGAN.")
    parser.add_argument("directory", type=str, help="Directory containing .pt files.")
    parser.add_argument("bigvgan_model_name", type=str, help="Name of the BigVGAN model to use.")
    args = parser.parse_args()
    
    process_directory(args.directory, args.bigvgan_model_name)
