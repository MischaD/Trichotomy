import argparse
import os
import torch.multiprocessing as mp
import torch
import torch.distributed as dist
from tqdm import tqdm
from src.data import get_distributed_image_dataloader, get_data_from_txt, get_data, get_data_from_folder
import os
import torch 
from ml_collections import ConfigDict
from cosmos_tokenizer.image_lib import ImageTokenizer
from torchvision.utils import save_image


def setup(rank, world_size, master_port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port) 
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def get_latent_model(model_name=None):
    # Load the VQ-VAE model
    if model_name is None: 
        # Download: https://huggingface.co/nvidia/Cosmos-Tokenizer-CI16x16 -- Step 2 
        model_name = "Cosmos-Tokenizer-CI16x16"

    print(f"Loading model: {model_name}")
    encoder = ImageTokenizer(checkpoint_enc=f'pretrained_ckpts/{model_name}/encoder.jit')
    decoder = ImageTokenizer(checkpoint_dec=f'pretrained_ckpts/{model_name}/decoder.jit')

    encoder.eval()
    decoder.eval()

    return encoder, decoder


def compute_latent_representation(images, encoder):
    (latent,) = encoder.encode(images) # 0 to 1 bchw
    return latent  

def compute_reconstruction(latents, decoder):
    reconstructed_tensor = decoder.decode(latents)
    return reconstructed_tensor 


def process(rank, world_size, file_list, config, save_path, skip_decoding=False, basedir=None):
    setup(rank, world_size, config.master_port)
    dataloader = get_distributed_image_dataloader(file_list, rank, world_size, config, base_name=basedir)

    encoder, decoder = get_latent_model()
    encoder = encoder.to(f"cuda:{rank}")
    decoder = decoder.to(f"cuda:{rank}")

    latents_save_path = os.path.join(save_path, "latents")
    recon_save_path = os.path.join(save_path, "reconstructed")

    # Only initialize tqdm for rank 0
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Rank {rank}/{world_size} computing latents", bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}")
    else:
        pbar = dataloader  # No progress bar for other ranks

    with torch.no_grad():
        for images, indices, paths in pbar:
            # images will now be a batch of 16 images, and paths will have 16 corresponding paths
            images = images.to(f"cuda:{rank}").to(torch.bfloat16)
            image_latents = compute_latent_representation(images, encoder)

            # Save each latent for every image in the batch
            for i, path in enumerate(paths):
                latent_save_path = os.path.join(latents_save_path, path + ".pt")
                os.makedirs(os.path.dirname(latent_save_path), exist_ok=True)
                torch.save(image_latents[i].cpu(), latent_save_path)

            if not skip_decoding: 
                rec = compute_reconstruction(image_latents, decoder)
                for i, path in enumerate(paths):
                    img_save_path = os.path.join(recon_save_path, path)
                    os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
                    save_image(rec[i].clip(0,1).cpu(), img_save_path)

    cleanup()


def main(config):
    world_size = torch.cuda.device_count()

    if config.filelist.endswith(".csv"):
        file_list, _ =  get_data(config)
    elif config.filelist.endswith(".txt"): 
        file_list, _ = get_data_from_txt(config)
    else: 
        file_list, _ =  get_data_from_folder(config.filelist)

    output_dir = config.output_path
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving latents to {output_dir}/latents and reconstructions to {output_dir}/reconstructed")
    

    mp.spawn(process, args=(world_size, file_list, config, output_dir, config.skip_decoding, config.basedir), nprocs=world_size, join=True)

    print(f"Latents saved to {output_dir}")


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("EXP_NAME", type=str, help="Path to Experiment results")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for DataLoader")
    parser.add_argument("--basedir", type=str, help="Path to basedir of dataset")
    parser.add_argument("--output_path", type=str, help="Output path. Default is data_dir/<model>_latents")
    parser.add_argument("--filelist", type=str, help="data dir or csv with paths to data")
    parser.add_argument("--skip_decoding", action="store_true", help="Dont decode images")
    parser.add_argument('--master_port', type=int, default=12344)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    args.compute_latent = ConfigDict()
    args.compute_latent.input_size = 512 
    args.compute_latent.batch_size = args.batch_size
    main(args)
