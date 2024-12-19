import argparse
import os
from beyondfid.log import logger
from utils import main_setup
import torch.multiprocessing as mp
import torch
import torch.distributed as dist
from tqdm import tqdm
from src.latent import compute_latent_representation, get_latent_model
from src.data import get_distributed_image_dataloader, get_data_from_txt, get_data, get_data_from_folder
import os
import torch 
from diffusers import AutoencoderKL


def setup(rank, world_size, master_port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port) 
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def get_latent_model(path=None):
    # Load the VQ-VAE model
    if path is None: 
        path = "stabilityai/stable-diffusion-2"
    model = AutoencoderKL.from_pretrained(path, subfolder="vae")
    model = model.to("cuda")
    model.eval()
    return model


def process(rank, world_size, file_list, model, config, save_path):
    setup(rank, world_size, config.master_port)

    dataloader = get_distributed_image_dataloader(file_list, rank, world_size, config)
    model = model.to(f"cuda:{rank}")
    model.eval()

    # Only initialize tqdm for rank 0
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Rank {rank}/{world_size} computing latents", bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}")
    else:
        pbar = dataloader  # No progress bar for other ranks

    with torch.no_grad():
        for images, indices, paths in pbar:
            # images will now be a batch of 16 images, and paths will have 16 corresponding paths
            images = images.to(f"cuda:{rank}")
            image_latents = compute_latent_representation(images, model, config.compute_latent.batch_size)

            # Save each latent for every image in the batch
            for i, path in enumerate(paths):
                img_save_path = os.path.join(save_path, path + ".pt")
                os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
                torch.save(image_latents[i].cpu(), img_save_path)
    cleanup()


def main(config):
    world_size = torch.cuda.device_count()

    if config.filelist.endswith(".csv"):
        file_list, _ =  get_data(config)
    elif config.filelist.endswith(".txt"): 
        file_list, _ = get_data_from_txt(config)
    else: 
        file_list, _ =  get_data_from_folder(config.filelist)

    latents_output_dir = config.get("output_path", os.path.join(os.path.dirname(config.filelist), "Latents"))
    os.makedirs(latents_output_dir, exist_ok=True)
    logger.info(f"Saving latent to {latents_output_dir}")

    model = get_latent_model(path=config.compute_latent.model_path)
    mp.spawn(process, args=(world_size, file_list, model, config, latents_output_dir), nprocs=world_size, join=True)

    print(f"Latents saved to {latents_output_dir}")


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("EXP_NAME", type=str, help="Path to Experiment results")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for DataLoader")
    parser.add_argument("--output_path", type=str, help="Output path. Default is data_dir/<model>_latents")
    parser.add_argument("--filelist", type=str, help="data dir or csv with paths to data")
    parser.add_argument('--master_port', type=int, default=12344)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = main_setup(args, name=os.path.basename(__file__).rstrip('.py'))
    config.compute_latent.batch_size = args.batch_size
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != config.dm_training.local_rank:
        config.dm_training.local_rank = env_local_rank
    config.debug = config.EXP_NAME.endswith("debug")
    config.master_port = args.master_port
    main(config)
