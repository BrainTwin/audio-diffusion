# based on https://github.com/CompVis/stable-diffusion/blob/main/main.py
import sys
print(sys.path)
import argparse
import os

import numpy as np
import pytorch_lightning as pl
# if getting the error: cannot import name '_compare_version' from 'torchmetrics.utilities.imports' 
# use this fix: https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/11648
import torch
print(f'torch cuda version is: {torch.version.cuda}')
print(f'torch cuda availability is: {torch.cuda.is_available()}')
import torchvision
from datasets import load_dataset, load_from_disk
from diffusers.pipelines.audio_diffusion import Mel
from librosa.util import normalize
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

# we want this script to first look into the local directory, rather than the installed audiodiffusion library
sys.path.insert(0, '/home/th716/rds/hpc-work/audio-diffusion/')
sys.path.insert(0, '/home/th716/audio-diffusion/')
print(f'new sys.path is: {sys.path}')
from audiodiffusion.utils import convert_ldm_to_hf_vae

sys.path.append('/home/th716/rds/hpc-work/audio-diffusion/stable-diffusion/')
from ldm.util import instantiate_from_config

print(sys.path)

class AudioDiffusion(Dataset):
    def __init__(self, model_id, channels=3, max_samples=None):
        super().__init__()
        self.channels = channels
        if os.path.exists(model_id):
            self.hf_dataset = load_from_disk(model_id)["train"]
        else:
            self.hf_dataset = load_dataset(model_id)["train"]
        if max_samples is not None:
            self.hf_dataset = self.hf_dataset.select(range(min(max_samples, len(self.hf_dataset))))

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        image = self.hf_dataset[idx]["image"]
        if self.channels == 3:
            image = image.convert("RGB")
        image = np.frombuffer(image.tobytes(), dtype="uint8").reshape((image.height, image.width, self.channels))
        image = (image / 255) * 2 - 1
        return {"image": image}



class AudioDiffusionDataModule(pl.LightningDataModule):
    def __init__(self, model_id, batch_size, channels, max_samples=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = AudioDiffusion(model_id=model_id, channels=channels, max_samples=max_samples)
        self.num_workers = 1

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)



class ImageLogger(Callback):
    def __init__(self, every=1000, hop_length=512, sample_rate=22050, n_fft=2048):
        super().__init__()
        self.every = every
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.n_fft = n_fft

    @rank_zero_only
    def log_images_and_audios(self, pl_module, batch):
        pl_module.eval()
        with torch.no_grad():
            images = pl_module.log_images(batch, split="train")
        pl_module.train()

        image_shape = next(iter(images.values())).shape
        channels = image_shape[1]
        mel = Mel(
            x_res=image_shape[2],
            y_res=image_shape[3],
            hop_length=self.hop_length,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
        )

        for k in images:
            images[k] = images[k].detach().cpu()
            images[k] = torch.clamp(images[k], -1.0, 1.0)
            images[k] = (images[k] + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = torchvision.utils.make_grid(images[k])

            tag = f"train/{k}"
            pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)

            images[k] = (images[k].numpy() * 255).round().astype("uint8").transpose(0, 2, 3, 1)
            for _, image in enumerate(images[k]):
                audio = mel.image_to_audio(
                    Image.fromarray(image, mode="RGB").convert("L")
                    if channels == 3
                    else Image.fromarray(image[:, :, 0])
                )
                pl_module.logger.experiment.add_audio(
                    tag + f"/{_}",
                    normalize(audio),
                    global_step=pl_module.global_step,
                    sample_rate=mel.get_sample_rate(),
                )
                

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx + 1) % self.every != 0:
            return
        self.log_images_and_audios(pl_module, batch)


class HFModelCheckpoint(ModelCheckpoint):
    def __init__(self, exp_name, ldm_config, hf_checkpoint, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ldm_config = ldm_config
        self.hf_checkpoint = hf_checkpoint
        self.sample_size = None
        # Customize the directory path using the experiment name
        self.dirpath = os.path.join(self.dirpath, exp_name)  # Append the experiment name to the directory path

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.sample_size is None:
            self.sample_size = list(batch["image"].shape[1:3])

    def on_train_epoch_end(self, trainer, pl_module):
        ldm_checkpoint = self._get_metric_interpolated_filepath_name({"epoch": trainer.current_epoch}, trainer)
        super().on_train_epoch_end(trainer, pl_module)
        self.ldm_config.model.params.ddconfig.resolution = self.sample_size
        convert_ldm_to_hf_vae(ldm_checkpoint, self.ldm_config, self.hf_checkpoint, self.sample_size)


def print_gpu_memory():
    allocated_memory = torch.cuda.memory_allocated()
    max_allocated_memory = torch.cuda.max_memory_allocated()
    print(f"Current memory allocated: {allocated_memory / (1024 ** 2):.2f} MB")
    print(f"Max memory allocated so far: {max_allocated_memory / (1024 ** 2):.2f} MB")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE using ldm.")
    parser.add_argument("-d", "--dataset_name", type=str, default=None)
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("-c", "--ldm_config_file", type=str, default="config/ldm_autoencoder_kl.yaml")
    parser.add_argument("--ldm_checkpoint_dir", type=str, default="models/ldm-autoencoder-kl")
    parser.add_argument("--hf_checkpoint_dir", type=str, default="models/autoencoder-kl")
    parser.add_argument("-r", "--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("-g", "--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--save_images_batches", type=int, default=1000)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to load from the dataset")
    parser.add_argument("--latent_dims", type=str, default="4,4", help="Latent space dimensions, e.g., '4,4' for 4x4 latent space")
    parser.add_argument("--resolution", type=int, default="256", help="We are assuming that the resolution is square, e.g. NxN.")
    parser.add_argument("--model_size", type=str, default="small", help="Determine the amount of channel multipliers across layers.")

    args = parser.parse_args()
    
    # setup config and args
    latent_dims = tuple(map(int, args.latent_dims.split(',')))
    exp_name = f'd_{args.dataset_name}_'

    config = OmegaConf.load(args.ldm_config_file)
    config.model.params.ddconfig.latent_resolution = latent_dims
    config.model.params.ddconfig.resolution = args.resolution
    
    downsampling_steps = int(np.log2(args.resolution / latent_dims[0]))  # log2 of how much we need to downsample
    if args.model_size == 'small':
        ch_multipliers = [2, 2, 4, 4, 4, 8, 8, 8, 8, 16, 16]
        
    elif args.model_size == 'medium':
        ch_multipliers = [2, 2, 4, 4, 8, 8, 16, 16, 32, 32, 64]
    elif args.model_size == 'large':
        ch_multipliers = [2, 4, 4, 8, 8, 16, 32, 32, 64, 64, 128]
        
        
    config.model.params.ddconfig.ch_mult = [1] + ch_multipliers[:downsampling_steps]


    # instantiate model and necessary objects    
    print('Printing memory usage BEFORE model instantiation')
    print_gpu_memory()
    
    model = instantiate_from_config(config.model)
    print('Printing memory usage AFTER model instantiation')
    print_gpu_memory()
    
    model.learning_rate = config.model.base_learning_rate
    data = AudioDiffusionDataModule(
        model_id=args.dataset_name,
        batch_size=args.batch_size,
        channels=config.model.params.ddconfig.in_channels,
        max_samples=args.max_samples
    )
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_config.accumulate_grad_batches = args.gradient_accumulation_steps
    trainer_opt = argparse.Namespace(**trainer_config)
    
    tb_logger = TensorBoardLogger(save_dir=args.ldm_checkpoint_dir, name='tensorboard_logs')
    
    gpu_stats = DeviceStatsMonitor()
    trainer = Trainer.from_argparse_args(
        trainer_opt,
        max_epochs=args.max_epochs,
        resume_from_checkpoint=args.resume_from_checkpoint,
        logger=tb_logger,
        callbacks=[
            ImageLogger(
                every=args.save_images_batches,
                hop_length=args.hop_length,
                sample_rate=args.sample_rate,
                n_fft=args.n_fft,
            ),
            HFModelCheckpoint(
                ldm_config=config,
                hf_checkpoint=args.hf_checkpoint_dir,
                dirpath=args.ldm_checkpoint_dir,
                filename="{epoch:06}",
                verbose=True,
                save_last=True,
                exp_name=exp_name,
            ),
            gpu_stats
        ],
    )
    trainer.fit(model, data)
