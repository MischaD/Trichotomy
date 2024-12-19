import argparse
import os
import pandas as pd
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
import torch 
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL
from src.latent import decode_latent_representation, get_latent_model
from src.data import get_data_from_txt, get_data, get_data_from_folder, get_dataloader
from torchvision.utils import save_image



def process(file_list, model, config, save_path):

    dataloader = get_dataloader(file_list, config)
    # Only initialize tqdm for rank 0
    pbar = tqdm(dataloader, desc=f"Reconstructing Images", bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}")

    with torch.no_grad():
        for latents, indices, paths in pbar:
            # images will now be a batch of 16 images, and paths will have 16 corresponding paths
            latents = latents.to(f"cuda:0")
            images = decode_latent_representation(latents, model)

            # Save each latent for every image in the batch
            for i, path in enumerate(paths):
                img_save_path = os.path.join(save_path, path)
                os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
                save_image(images[i].clip(0,1).cpu(), img_save_path)


def main(config):
    if config.filelist.endswith(".csv"):
        file_list, _ =  get_data(config)
    elif config.filelist.endswith(".txt"): 
        file_list, _ = get_data_from_txt(config)
    else: 
        file_list, _ =  get_data_from_folder(config.filelist)

    latents_output_dir = config.output_path
    os.makedirs(latents_output_dir, exist_ok=True)
    print(f"Saving latent to {latents_output_dir}")

    model = get_latent_model(path=config.latent_model_path)
    process(file_list, model, config, latents_output_dir)

    print(f"Latents saved to {latents_output_dir}")


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for DataLoader")
    parser.add_argument("--basedir", type=str, help="Path to basedir of latent dataset")
    parser.add_argument("--latent_model_path", type=str, default="stabilityai/stable-diffusion-2", help="Huggingface Path to vae")
    parser.add_argument("--output_path", type=str, help="Output path. Default is data_dir/<model>_latents")
    parser.add_argument("--filelist", type=str, help="data dir or csv with paths to data")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
