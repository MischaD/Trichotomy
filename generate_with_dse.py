import argparse
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
from einops import repeat
from tqdm import tqdm
from pprint import pprint
from src.utils import class_labels
from src.dse import DiADMSampleEvaluator
from src.diffusion.generation import get_image_generation_model, ImageIterableDSE
from edm2.training.dataset import LatentDataset
from edm2.generate_images import edm_sampler


def setup(rank, world_size, master_port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port) 
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def check_for_already_generated(outdir, filelist, basedir):
    """
    For each line in filelist, checks if the image path exists within basedir.
    If it does not exist, writes that line to a temporary file and returns the file's path.

    Args:
        outdir (str): Directory to check if it exists.
        filelist (str): Path to the input file list.
        basedir (str): Base directory where the image files should be located.

    Returns:
        str: Path to the temporary file containing non-existing image lines or None if all exist.
    """
    missing_imgs_idx = []
    idx = 0
    # Read filelist and check each image path
    with open(filelist, "r") as fp:
        for line in fp:
            if line.strip():  # Ignore empty lines
                parts = line.split()
                image_path = os.path.join(outdir, parts[0])
                if not os.path.exists(image_path):
                    missing_imgs_idx.append(idx)
                idx += 1

    if len(missing_imgs_idx) != idx: 
        print(f"Already found {idx - len(missing_imgs_idx)} images. Script will generate the remaining: {len(missing_imgs_idx)}")
    return missing_imgs_idx 


def generate(rank, world_size, master_port, outdir, filelist, model_kwargs, n_per_index, ds_kwargs, sampler_kwargs, clf_path, priv_path, missing_imgs_idx):
    setup(rank, world_size, master_port)

    train_ds = LatentDataset(filelist_txt=filelist, basedir=ds_kwargs["basedir"], cond_mode=ds_kwargs["cond_mode"], load_to_memory=False)

    indices = torch.tensor(missing_imgs_idx)
    indices = repeat(indices, "l -> b l", b=n_per_index)
    indices = indices.transpose(0, 1).flatten()

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    model_kwargs["device"] = device
    net, gnet, encoder = get_image_generation_model(**model_kwargs)
    dse = DiADMSampleEvaluator(device, clf_path=clf_path, priv_path=priv_path)

    # split_indices_per_gpu
    imgs_per_gpu = (len(train_ds) // world_size) * n_per_index
    indices_start = imgs_per_gpu * rank
    indices_stop = imgs_per_gpu * (rank + 1) if rank != world_size -1 else len(train_ds) * n_per_index
    indices_per_gpu = indices[indices_start:indices_stop]

    image_iter = ImageIterableDSE(train_ds=train_ds, 
                                  indices=indices_per_gpu, 
                                  device=device, 
                                  net=net, 
                                  sampler_fn=edm_sampler, 
                                  gnet=gnet, 
                                  encoder=encoder,
                                  outdir=outdir, 
                                  max_batch_size=n_per_index,
                                  dse=dse,
                                  sampler_kwargs=sampler_kwargs)

    # Only show tqdm progress bar in rank 0
    if rank == 0:
        pbar = tqdm(total=len(image_iter), unit='batch', desc=f"GPU {rank}: Generating images")
        for _ in image_iter:
            pbar.update(1)
        pbar.close()
    else:
        for _ in image_iter:
            pass  # No progress bar for other ranks

    cleanup()


def parse_args():
    parser = argparse.ArgumentParser(description="Run image generation model with configurable parameters.")
    parser.add_argument("--n_per_index", type=int, default=4, help="Batch size and sampling factor.")
    parser.add_argument("--filelist", type=str, default="/vol/ideadata/ed52egek/pycharm/trichotomy/datasets/eight_cxr8_train.txt", help="Path to the filelist.")
    parser.add_argument("--target_dir", type=str, default="diadm_train_with_dse", help="Target directory for generated images.")
    parser.add_argument("--basedir", type=str, help="Basedir of dataset")
    parser.add_argument("--mode", type=str, required=True, help="Mode for the model configuration.")
    parser.add_argument("--guidance", type=float, default=1.4, help="Guidance parameter.")
    parser.add_argument("--pseudo_cond_feature_extractor", help='Feature extractor for the pseudocon model. Precompute using beyondfid.', default="inception")
    parser.add_argument("--model_weights", type=str, default="/vol/ideadata/ed52egek/pycharm/trichotomy/importantmodels/cxr8_diffusionmodels/baseline-runs/cxr8_pseudocond/training-state-0050331.pt", help="Path to model weights.")
    parser.add_argument("--gmodel_weights", type=str, default="/vol/ideadata/ed52egek/pycharm/trichotomy/importantmodels/cxr8_diffusionmodels/baseline-runs/cxr8_uncond/training-state-0008388.pt", help="Path to guidance model weights.")
    parser.add_argument("--path_net", type=str, default="/vol/ideadata/ed52egek/pycharm/trichotomy/importantmodels/cxr8_diffusionmodels/baseline-runs/cxr8_pseudocond/network-snapshot-0050331-0.100.pkl", help="Path to network snapshot.")
    parser.add_argument("--path_gnet", type=str, default="/vol/ideadata/ed52egek/pycharm/trichotomy/importantmodels/cxr8_diffusionmodels/baseline-runs/cxr8_uncond/network-snapshot-0008388-0.050.pkl", help="Path to guidance network snapshot.")
    parser.add_argument("--clf_path", type=str, default="/vol/ideadata/ed52egek/pycharm/trichotomy/importantmodels/results_chexnet_real/saved_models_cxr8/m-05122024-131940.pth.tar", help="Path to classifier network.")
    parser.add_argument("--priv_path", type=str, default="/vol/ideadata/ed52egek/pycharm/trichotomy/privacy/archive/Siamese_ResNet50_allcxr/Siamese_ResNet50_allcxr_checkpoint.pth", help="Path to classifier network.")
    parser.add_argument('--master_port', type=int, default=12344)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    world_size = torch.cuda.device_count()

    kwargs = {
        "DiADM": {
            "autoguidance": False,# only works if second model is conditional
            "guidance": args.guidance,
            "model_kwargs": {
                "model_weights": args.model_weights,
                "gmodel_weights": args.gmodel_weights,
                "path_net": args.path_net,
                "path_gnet": args.path_gnet,
            },
            "priv_path": args.priv_path,
            "clf_path": args.clf_path,
            "ds_kwargs": {
                "cond_mode": "pseudocond",
                "basedir": args.basedir,
                "pseudo_cond_feature_extractor": args.pseudo_cond_feature_extractor,
            }
        }
    }

    model_kwargs = kwargs[args.mode]["model_kwargs"]
    ds_kwargs = kwargs[args.mode]["ds_kwargs"]
    model_kwargs["name"] = args.mode
    
    print("=" * 80)
    print("Model kwargs:")
    pprint(model_kwargs)
    print("Dataset kwargs:")
    pprint(ds_kwargs)
    print("=" * 80)
    

    outdir = f"./{args.target_dir}/"
    print(f"Saving images to {outdir}")

    sampler_kwargs = {
        "autoguidance": kwargs[args.mode]["autoguidance"], 
        "guidance": kwargs[args.mode]["guidance"], 
    }

    print("Sampler kwargs")
    pprint(sampler_kwargs)
    

    #generate(outdir, args.filelist, net, gnet, encoder, args.n_per_index, ds_kwargs, sampler_kwargs)
    print(f"Saved images to {outdir}")


    # check if we have already generated some images, if so generates a new .txt filelist 
    missing_imgs_idx = check_for_already_generated(outdir=outdir, filelist=args.filelist, basedir=ds_kwargs["basedir"])

    mp.spawn(generate, args=(world_size,  args.master_port, outdir, args.filelist, model_kwargs, args.n_per_index, ds_kwargs, sampler_kwargs,  kwargs[args.mode]["clf_path"], kwargs[args.mode]["priv_path"], missing_imgs_idx), nprocs=world_size, join=True)

