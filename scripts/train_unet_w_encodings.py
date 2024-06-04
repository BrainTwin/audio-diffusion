# based on https://github.com/huggingface/diffusers/blob/main/examples/train_unconditional.py

import argparse
import io
import os
import pickle
import random
from pathlib import Path
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset, load_from_disk
from diffusers import (AutoencoderKL, DDIMScheduler, DDPMScheduler,
                       UNet2DConditionModel, UNet2DModel, UNet1DModel)
from diffusers.optimization import get_scheduler
from diffusers.pipelines.audio_diffusion import Mel
from diffusers.training_utils import EMAModel # currently using 0.24.0 diffusers library
from huggingface_hub import HfFolder, Repository, whoami
import librosa
from librosa.util import normalize
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import sys
sys.path.insert(0, '/home/th716/rds/hpc-work/audio-diffusion/')
print(sys.path)
from audiodiffusion.pipeline_audio_diffusion import AudioDiffusionPipeline


logger = get_logger(__name__)

def load_image_from_bytes(image_bytes):
    return Image.open(io.BytesIO(image_bytes))

def denoise_waveforms(model, noisy_waveforms, num_inference_steps, scheduler):
    model.eval()
    with torch.no_grad():
        for t in tqdm(range(num_inference_steps), desc='Running inference'):
            # Get the model prediction
            noise_pred = model(noisy_waveforms, t).sample
            # Use the scheduler to compute the denoised waveform
            noisy_waveforms = scheduler.step(noise_pred, t, noisy_waveforms).prev_sample
    return noisy_waveforms

def get_full_repo_name(model_id: str,
                       organization: Optional[str] = None,
                       token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

class AudioDataset(Dataset):
    def __init__(self, directory, target_sample_rate=22050, chunk_length=65536):
        self.files = list(Path(directory).glob("*.wav"))
        self.sample_rate = target_sample_rate
        self.chunk_length = chunk_length
        self.data = []
        self._prepare_dataset()

    def _prepare_dataset(self):
        for file_path in tqdm(self.files, desc='Splitting files for chunked waveform data processing'):
            audio, sample_rate = librosa.load(file_path, sr=None)
            if audio.ndim == 2:
                audio = librosa.to_mono(audio)
            if sample_rate != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)
            audio = librosa.util.normalize(audio)
            self._split_and_store_audio(audio)

    def _split_and_store_audio(self, audio):
        total_length = len(audio)
        num_chunks = total_length // self.chunk_length
        for i in range(num_chunks):
            chunk = audio[i * self.chunk_length:(i + 1) * self.chunk_length]
            self.data.append(chunk)
        if total_length % self.chunk_length != 0:
            last_chunk = audio[num_chunks * self.chunk_length:]
            pad_width = self.chunk_length - len(last_chunk)
            last_chunk = np.pad(last_chunk, (0, pad_width), 'constant')
            self.data.append(last_chunk)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_chunk = self.data[idx]
        return torch.tensor(audio_chunk, dtype=torch.float32)

def main(args):
    output_dir = os.environ.get("SM_MODEL_DIR", None) or args.output_dir
    logging_dir = os.path.join(output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=logging_dir,
    )

    # Conditional initialization of Mel
    mel = None
    if not args.use_waveform:
        if args.dataset_name is not None:
            if os.path.exists(args.dataset_name):
                dataset = load_from_disk(
                    args.dataset_name,
                    storage_options=args.dataset_config_name)["train"]
            else:
                dataset = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    cache_dir=args.cache_dir,
                    use_auth_token=True if args.use_auth_token else None,
                    split="train",
                )
        else:
            dataset = load_dataset(
                "imagefolder",
                data_dir=args.train_data_dir,
                cache_dir=args.cache_dir,
                split="train",
            )
            
        # Determine image resolution
        image = load_image_from_bytes(dataset[0]["image"]["bytes"])
        resolution = image.height, image.width

        augmentations = Compose([
            ToTensor(),
            Normalize([0.5], [0.5]),
        ])
        
        use_encodings = True
        
        def transforms(examples):
            try:
                images = [augmentations(load_image_from_bytes(image["bytes"])) for image in examples["image"]]

            except TypeError as e:
                logger.error(f"Expected a dictionary with bytes, but got: {type(image)}. Error: {str(e)}")
                raise

            if 'encoding' in examples:
                encoding = [torch.tensor(enc, dtype=torch.float32) for enc in examples['encoding']]
                return {"input": images, "encoding": encoding}
            
            return {"input": images}

        dataset.set_transform(transforms)
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.train_batch_size, shuffle=True)
    # LOAD RAW WAVEFORMS
    else:
        train_dataset = AudioDataset(args.train_data_dir, chunk_length=args.waveform_resolution)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    vqvae = None
    if args.vae is not None:
        try:
            vqvae = AutoencoderKL.from_pretrained(args.vae)
        except EnvironmentError:
            vqvae = AudioDiffusionPipeline.from_pretrained(args.vae).vqvae
        # Determine latent resolution
        with torch.no_grad():
            latent_resolution = vqvae.encode(
                torch.zeros((1, 1) +
                            resolution)).latent_dist.sample().shape[2:]

    if args.from_pretrained is not None:
        pipeline = AudioDiffusionPipeline.from_pretrained(args.from_pretrained)
        if not args.use_waveform:
            mel = pipeline.mel
        model = pipeline.unet
        if hasattr(pipeline, "vqvae"):
            vqvae = pipeline.vqvae
    else:
        if args.model_size == 'large':
            model = UNet2DConditionModel(
                sample_size=resolution if vqvae is None else latent_resolution,
                in_channels=1
                if vqvae is None else vqvae.config["latent_channels"],
                out_channels=1
                if vqvae is None else vqvae.config["latent_channels"],
                layers_per_block=2,
                block_out_channels=(128, 256, 256, 512, 512, 512, 1024),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
                cross_attention_dim=list(encodings.values())[0].shape[-1],
            )
        elif args.model_size == 'small':
            model = UNet2DConditionModel(
                sample_size=resolution if vqvae is None else latent_resolution,
                in_channels=1
                if vqvae is None else vqvae.config["latent_channels"],
                out_channels=1
                if vqvae is None else vqvae.config["latent_channels"],
                layers_per_block=2,
                block_out_channels=(128, 256, 256, 256, 512, 512),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "CrossAttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "CrossAttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
                cross_attention_dim=list(encodings.values())[0].shape[-1],
            ) 

    # Initialize schedulers
    if args.train_scheduler == "ddpm":
        train_noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.num_train_steps)
    else:
        train_noise_scheduler = DDIMScheduler(
            num_train_timesteps=args.num_train_steps)
        
    if args.test_scheduler == "ddpm":
        test_noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.num_inference_steps)
    else:
        test_noise_scheduler = DDIMScheduler(
            num_train_timesteps=args.num_inference_steps)
        
    test_noise_scheduler.set_timesteps(args.num_inference_steps, device=accelerator.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs) //
        args.gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)

    ema_model = EMAModel(
        getattr(model, "module", model),
        inv_gamma=args.ema_inv_gamma,
        power=args.ema_power,
        max_value=args.ema_max_decay,
    )

    if args.push_to_hub:
        if args.hub_model_id is None:
            repo_name = get_full_repo_name(Path(output_dir).name,
                                           token=args.hub_token)
        else:
            repo_name = args.hub_model_id
        repo = Repository(output_dir, clone_from=repo_name)

    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    if not args.use_waveform:
        mel = Mel(
            x_res=resolution[1],
            y_res=resolution[0],
            hop_length=args.hop_length,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
        )

    global_step = 0
    total_start_time = time.time()
    
    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        progress_bar = tqdm(total=len(train_dataloader),
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        if epoch < args.start_epoch:
            for step in range(len(train_dataloader)):
                optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)
                global_step += 1
            if epoch == args.start_epoch - 1 and args.use_ema:
                ema_model.optimization_step = global_step
            continue

        model.train()
        for step, batch in enumerate(train_dataloader):
            if args.use_waveform:
                clean_waveforms = batch
                clean_waveforms = clean_waveforms.unsqueeze(1)  # shape: [batch_size, 1, sample_size]
                clean_images = None
            else:
                clean_images = batch["input"]
                clean_waveforms = None

            if vqvae is not None and clean_images is not None:
                vqvae.to(clean_images.device)
                with torch.no_grad():
                    clean_images = vqvae.encode(
                        clean_images).latent_dist.sample()
                clean_images = clean_images * 0.18215

            noise = torch.randn(clean_images.shape if clean_images is not None else clean_waveforms.shape).to(clean_images.device if clean_images is not None else clean_waveforms.device)
            bsz = clean_images.shape[0] if clean_images is not None else clean_waveforms.shape[0]
            timesteps = torch.randint(
                0,
                train_noise_scheduler.config.num_train_timesteps,
                (bsz, ),
                device=noise.device,
            ).long()

            noisy_images = train_noise_scheduler.add_noise(clean_images, noise, timesteps) if clean_images is not None else None
            noisy_waveforms = train_noise_scheduler.add_noise(clean_waveforms, noise, timesteps) if clean_waveforms is not None else None

            with accelerator.accumulate(model):
                if use_encodings:
                    encodings = batch["encoding"]
                    # Move encodings to the correct device
                    encodings = encodings.to(noise.device)
                    if len(encodings.shape) == 2:
                        encodings = encodings.unsqueeze(1)  # Add sequence_length dimension
                    noise_pred = model(noisy_images, timesteps, encodings)
                else:
                    noise_pred = model(noisy_images if noisy_images is not None else noisy_waveforms, timesteps)
                loss = F.mse_loss(noise_pred["sample"], noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                if args.use_ema:
                    ema_model.step(model)
                optimizer.zero_grad()


            progress_bar.update(1)
            global_step += 1

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            if args.use_ema:
                logs["ema_decay"] = ema_model.decay
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            
            if accelerator.is_main_process:
                if (global_step in args.save_model_steps
                        or global_step in args.save_images_steps
                        or epoch == args.num_epochs - 1
                        or global_step >= args.max_training_num_steps):
                    unet = accelerator.unwrap_model(model)
                    if args.use_ema:
                        ema_model.copy_to(unet.parameters())
                    if args.use_waveform:
                        torch.save(unet.state_dict(), os.path.join(output_dir, f"unet1dmodel_step_{global_step}.pth"))
                    else:
                        pipeline = AudioDiffusionPipeline(
                            vqvae=vqvae,
                            unet=unet,
                            mel=mel,
                            scheduler=train_noise_scheduler,
                        )
                        pipeline.save_pretrained(os.path.join(output_dir, f"model_step_{global_step}"))

                    if args.push_to_hub:
                        repo.push_to_hub(
                            commit_message=f"global_step {global_step}",
                            blocking=False,
                            auto_lfs_prune=True,
                        )

                if global_step in args.save_images_steps \
                    or global_step >= args.max_training_num_steps:
                    generator = torch.Generator(device=noise.device).manual_seed(42)

                    if use_encodings:
                        random.seed(42)
                        with open(args.path_to_encodings_pickle, 'rb') as f:
                            encoding = pickle.load(f)
                        encoding = torch.tensor(encoding).to(noise.device)

                        # Adjust the encoding tensor based on eval_batch_size
                        if args.eval_batch_size < encoding.shape[0]:
                            encoding = encoding[:args.eval_batch_size]
                        elif args.eval_batch_size > encoding.shape[0]:
                            repeats = args.eval_batch_size // encoding.shape[0]
                            remainder = args.eval_batch_size % encoding.shape[0]
                            encoding = encoding.repeat(repeats, 1, 1)
                            if remainder > 0:
                                encoding = torch.cat((encoding, encoding[:remainder]), dim=0)
                    else:
                        encoding = None

                    if args.use_waveform:
                        noise_shape = (args.eval_batch_size, 1, args.waveform_resolution)
                        noisy_waveforms = torch.randn(noise_shape, device=accelerator.device)  # Use accelerator.device to ensure correct device
                        generated_waveforms = denoise_waveforms(unet, noisy_waveforms, args.num_inference_steps, test_noise_scheduler)
                        
                        for idx, audio in enumerate(generated_waveforms):
                            accelerator.trackers[0].writer.add_audio(
                                f"test_audio_{idx}",
                                normalize(audio.cpu().numpy()),
                                global_step,
                                sample_rate=args.sample_rate,
                            )
                    else:
                        images, (sample_rate, audios) = pipeline(
                            generator=generator,
                            batch_size=args.eval_batch_size,
                            return_dict=False,
                            encoding=encoding,
                            steps=args.num_inference_steps
                        )

                        images = np.array([
                            np.frombuffer(image.tobytes(), dtype="uint8").reshape(
                                (len(image.getbands()), image.height, image.width))
                            for image in images
                        ])
                        accelerator.trackers[0].writer.add_images(
                            "test_samples", images, global_step)
                        for idx, audio in enumerate(audios):
                            accelerator.trackers[0].writer.add_audio(
                                f"test_audio_{idx}",
                                normalize(audio),
                                global_step,
                                sample_rate=sample_rate,
                            )
                accelerator.wait_for_everyone()

                if global_step >= args.max_training_num_steps:
                    accelerator.end_training()
                    return

            
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_training_time = epoch_end_time - total_start_time
        training_time_per_step = epoch_duration / len(train_dataloader) if train_dataloader else 0
        logs.update({
            "total_training_time": total_training_time,
            "training_time_per_step": training_time_per_step,
        })
        accelerator.log(logs, step=global_step)    
            
        progress_bar.close()

        accelerator.wait_for_everyone()

    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data.",
    )
    parser.add_argument("--use_waveform", type=bool, default=False)
    parser.add_argument("--waveform_resolution", type=int, default=65536)
    
    parser.add_argument("--path_to_encodings_pickle", type=str, default="/home/th716/audio-diffusion/cache/spotify_sleep_dataset/encodings.pkl")
    parser.add_argument("--model_size", type=str, default='small')
    
    parser.add_argument("--output_dir", type=str, default="ddpm-model-64")
    parser.add_argument("--overwrite_output_dir", type=bool, default=False)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--max_training_num_steps", type=int, default=10000)
    parser.add_argument("--save_images_steps", type=int, nargs='+', default=[10000, 25000, 100000, 200000])
    parser.add_argument("--save_model_steps", type=int, nargs='+', default=[10000, 25000, 100000, 200000])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--use_ema", type=bool, default=True)
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0)
    parser.add_argument("--ema_power", type=float, default=3 / 4)
    parser.add_argument("--ema_max_decay", type=float, default=0.9999)
    parser.add_argument("--push_to_hub", type=bool, default=False)
    parser.add_argument("--use_auth_token", type=bool, default=False)
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_private_repo", type=bool, default=False)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."),
    )
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--from_pretrained", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--num_train_steps", type=int, default=1000)
    parser.add_argument("--num_inference_steps", type=int, default=5, help="Number of timesteps for inference noise scheduling.")
    parser.add_argument("--train_scheduler", type=str, default="ddpm", help="ddpm or ddim")
    parser.add_argument("--test_scheduler", type=str, default="ddpm", help="ddpm or ddim")
    parser.add_argument(
        "--vae",
        type=str,
        default=None,
        help="pretrained VAE model for latent diffusion",
    )
    args = parser.parse_args()
    
    print(args)
    
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError(
            "You must specify either a dataset name from the hub or a train data directory."
        )

    main(args)
