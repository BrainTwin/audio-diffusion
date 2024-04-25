import argparse
import os
import torch
from pathlib import Path
import time
import torchaudio
from torchvision.utils import save_image
from tqdm import tqdm
from diffusers import AudioDiffusionPipeline
from diffusers import (AutoencoderKL, DDIMScheduler, DDPMScheduler,
                       UNet2DConditionModel, UNet2DModel)
from accelerate import Accelerator

def main(args):
    accelerator = Accelerator()

    model_path = args.model_path
    output_path = os.path.join(model_path, 'samples')
    tag = f'sch_{args.scheduler}_nisteps_{args.num_inference_steps}'
    images_path = os.path.join(output_path, 'images', tag)
    audios_path = os.path.join(output_path, 'audio', tag)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(audios_path, exist_ok=True)

    generator = torch.Generator(device='cuda').manual_seed(args.seed)
    
    if args.scheduler == "ddpm":
        scheduler = DDPMScheduler(
            num_train_timesteps=args.num_inference_steps)
        
    else:
        scheduler = DDIMScheduler(
            num_train_timesteps=args.num_inference_steps) 
        
    # Load the pretrained model
    pipeline = AudioDiffusionPipeline.from_pretrained(
        model_path,
        scheduler=scheduler
    )
    
    # Prepare the model for generation
    model = pipeline.unet
    model = accelerator.prepare(model)
    model.eval()
        
    # Calculate number of batches
    num_batches = (args.num_images + args.eval_batch_size - 1) // args.eval_batch_size

    # Generate and save images and audio
    generated_count = 0
    for batch_idx in tqdm(range(num_batches), desc=f'Saving images and audio into folder: {output_path}'):
        batch_size = min(args.eval_batch_size, args.num_images - generated_count)
        
        with torch.no_grad():
            images, (sample_rate, audios) = pipeline(
                generator=generator, 
                batch_size=args.eval_batch_size,
                return_dict=False, 
                steps=args.num_inference_steps,
            )

        for i in range(batch_size):
            image_index = generated_count + i + 1
            image_path = os.path.join(images_path, f'image_{image_index}.png')
            audio_path = os.path.join(audios_path, f'audio_{image_index}.wav')

            # Save image
            images[i].save(image_path)

            # Save audio
            audio_tensor = torch.tensor(audios[i]).unsqueeze(0)  # Convert numpy array to tensor and add batch dimension
            torchaudio.save(audio_path, audio_tensor, sample_rate)

        generated_count += batch_size

    print(f"All images and audios have been generated and saved to {output_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using a trained model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model directory.")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to generate.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps for generating images.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for generating images.")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Specify size of evaluation batch size.")
    parser.add_argument("--scheduler", type=str, default='ddim', choices=['ddim', 'ddpm'], help="Specify which scheduler to use, can be either ddim or ddpm.")

    args = parser.parse_args()
    main(args)
