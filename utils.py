import pytorch_lightning as pl
from enum import Enum
from einops import repeat
import argparse
from importlib.machinery import SourceFileLoader
import importlib
import torch
import numpy as np
import pandas as pd
from einops import rearrange
from scipy import ndimage
import torchvision
import matplotlib.pyplot as plt
from log import logger, log_experiment
from log import formatter as log_formatter
import os
import json
import shutil
import datetime
import logging
import hashlib


def hash_dataset_path(dataset_root_dir, img_list):
    """Takes a list of paths and joins it to a large string - then uses it as hash input stringuses it filename for the entire datsets for quicker loading"""
    name = "".join([x for x in img_list])
    name = hashlib.sha1(name.encode("utf-8")).hexdigest()
    return os.path.join(dataset_root_dir, "hashdata_" + name)


def data_loader(fn):
    """
    Decorator to handle the deprecation of data_loader from 0.7
    :param fn: User defined data loader function
    :return: A wrapper for the data_loader function
    """

    def func_wrapper(self):
        try: # Works for version 0.6.0
            return pl.data_loader(fn)(self)

        except: # Works for version > 0.6.0
            return fn(self)

    return func_wrapper


def make_exp_config(exp_file):
    if exp_file.endswith(".json"): 
        from utils import make_exp_config, json_to_dict 
        import ml_collections
        config = ml_collections.ConfigDict(json_to_dict(exp_file))
        exp_name = exp_file.split('/')[-1].rstrip('.json')
        config.name = exp_name
        return ml_collections.ConfigDict({"config": config})
        
    # get path to experiment
    exp_name = exp_file.split('/')[-1].rstrip('.py')

    # import experiment configuration
    exp_config = SourceFileLoader(exp_name, exp_file).load_module()
    exp_config.name = exp_name
    return exp_config


def resize_to(img, tosize):
    assert img.ndim == 4
    b, c, h, w = img.size()
    max_size = max(h, w)

    zoom_factor = tosize / max_size

    return torch.tensor(ndimage.zoom(img, (1, 1, zoom_factor,zoom_factor)))


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def img_to_viz(img):
    img = rearrange(img, "1 c h w -> h w c")
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
    img = np.array(((img + 1) * 127.5), np.uint8)
    return img


def collate_batch(batch):
    # make list of dirs to dirs of lists with batchlen
    batched_data = {}
    for data in batch:
        # label could be img, label, path, etc
        for key, value in data.items():
            if batched_data.get(key) is None:
                batched_data[key] = []
            if isinstance(value, list): 
                if isinstance(value[0], torch.Tensor):
                    # allows dataloader to return list of values
                    for i in range(len(value)):
                        batched_data[key].append(value[i])
            else: 
                batched_data[key].append(value)


    # cast to torch.tensor
    for key, value in batched_data.items():
        if isinstance(value[0],torch.Tensor):
            if value[0].size()[0] != 1:
                for i in range(len(value)):
                    value[i] = value[i][None,...]
            # check if concatenatable
            if all([value[0].size() == value[i].size() for i in range(len(value))]):
                batched_data[key] = torch.concat(batched_data[key])
    return batched_data


def viz_array(x):
    # 1 x c x h x w
    # c x h x w
    # h x w x c
    from einops import rearrange
    import matplotlib.pyplot as plt
    if not isinstance(x, torch.Tensor):
        x = torch.Tensor(x)
    x = x.float()
    x = x.squeeze()
    x = x.detach().cpu()
    x = (x - x.min()) / (x.max() - x.min())
    if x.ndim == 3:
        if x.size()[-1] != 3:
            x = rearrange(x, "c h w -> h w c")
        plt.imshow(x)
    else:
        #ndim == 2
        plt.imshow(x, cmap="Greys_r")
    plt.show()

def safe_viz_array(x, path): 
    # 1 x c x h x w
    # c x h x w
    # h x w x c
    from einops import rearrange
    import matplotlib.pyplot as plt
    if not isinstance(x, torch.Tensor):
        x = torch.Tensor(x)
    x = x.float()
    x = x.squeeze()
    x = x.detach().cpu()
    x = (x - x.min()) / (x.max() - x.min())
    if x.ndim == 3:
        if x.size()[-1] != 3:
            x = rearrange(x, "c h w -> h w c")
        plt.imshow(x)
    else:
        #ndim == 2
        plt.imshow(x, cmap="Greys_r")
    plt.savefig(os.path.abspath(path))
    logger.info(f"Saved to {path}")
    plt.show()


def load_scoresde_config(path):
    exp_config = SourceFileLoader("config", path).load_module()
    return exp_config.get_config()


def main_setup(args, name=__file__):
    config = make_exp_config(args.EXP_PATH).config
    for key, value in vars(args).items():
        if value is not None:
            keys = key.split(".")
            if len(keys) == 1:
                key = keys
                setattr(config, keys[0], value)
            else:
                # keys with more depth
                cfg_key = config
                for i in range(len(keys) - 1):
                    cfg_key = getattr(cfg_key, keys[i])
                setattr(cfg_key, keys[-1], value)
            logger.info(f"Overwriting exp file key {key} with: {value}")

    if not hasattr(config, "log_dir"):
        str_tim = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        log_dir = os.path.join(os.path.abspath("."), "log", args.EXP_NAME)
        trial = 0
        while os.path.exists(os.path.join(log_dir, f"{trial:04}_"+ name)): 
            trial += 1
        setattr(config, "log_dir",  os.path.join(log_dir,  f"{trial:04}_"+ name))
    else:
        # /vol/ideadata/ed52egek/pycharm/privacy/log/score_sde/2023-04-13T21-35-52
        config.EXP_NAME = config.log_dir.split("/")[-2] # overwrite exp name if log dir is defined

    log_dir = config.log_dir
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, f'{str_tim}.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.debug("="*30 + f"Running {os.path.basename(name)}" + "="*30)
    logger.debug(f"Logging to {log_dir}")

    # make log dir (same as the one for the console log)
    log_dir = os.path.join(os.path.dirname(file_handler.baseFilename))
    setattr(config, "log_dir", log_dir)
    logger.info(f"Log dir: {log_dir}")
    logger.debug(f"Current file: {__file__}")
    logger.debug(f"config")
    log_experiment(logger, args)
    # log config
    with open(os.path.join(config.log_dir, "config.json"), "w") as fp: 
        json.dump(config.to_dict(), fp, indent=6)
    return config


def repeat_channels(x):
    if x.size()[1] == 1:
        return repeat(x, "b 1 h w -> b 3 h w")
    return x


def inverse_scaler(x):
    return (x + 1.) / 2.

def to_float32(x):
    x = x.to(torch.float32)
    x = x / 255.
    return x


def to_uint8(x):
    x = x * 255.
    x = x.to(torch.uint8)
    return x


def data_scaler(x):
    """Data normalizer. Assume data are always in [0, 1]."""
    return x * 2. - 1.


def create_batches(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i+batch_size]

def dict_to_json(dct, json_path): 
    with open(json_path, 'w') as json_file:
        json.dump(dct, json_file, indent=4)
    return

def json_to_dict(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def save_copy_checkpoint(src_path, tgt_path, log_logdir=None, log_wandb=None):
    if src_path == "": 
        logger.info(f"Src path for save checkpoint is empty. No checkpoint to copy to {tgt_path}")
        return 

    os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
    if not os.path.exists(tgt_path):
        logger.info(f"Save best checkpoint to:{tgt_path}")
        shutil.copy(src_path, tgt_path)
    else:
        out_dir = os.path.dirname(tgt_path)
        extension = os.path.basename(tgt_path)
        i = 1
        while os.path.exists(os.path.join(out_dir, '{}_{}'.format(i, extension))):
            i += 1
        new_tgt_path = os.path.join(out_dir, '{}_{}'.format(i, extension))
        logger.info(f"Best path {tgt_path} already exists")
        logger.info(f"Copying old checkpoint to {new_tgt_path} as backup")
        shutil.copy(tgt_path, new_tgt_path)
        logger.info(f"Saving new checkpoint to {tgt_path}")
        shutil.copy(src_path, tgt_path)

    if log_logdir is not None:
        # some debug information
        base_path = os.path.dirname(tgt_path)
        extension = os.path.basename(tgt_path)
        with open(os.path.join(base_path, "." + extension + ".log"), "w", encoding="utf-8") as fp:
            fp.write(f"{tgt_path} comes from {log_logdir}\n")
            fp.write(f"wandb:{log_wandb}\n")



def update_matplotlib_font(fontsize=11, fontsize_ticks=8, tex=True):
    import matplotlib.pyplot as plt
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": tex,
        "font.family": "serif",
        # Use 11pt font in plots, to match 11pt font in document
        "axes.labelsize": fontsize,
        "font.size": fontsize,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": fontsize_ticks,
        "xtick.labelsize": fontsize_ticks,
        "ytick.labelsize": fontsize_ticks
    }
    plt.rcParams.update(tex_fonts)


def set_size(width, fraction=1, subplots=(1, 1), ratio= (5**.5 - 1) / 2):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "MICCAI":
        width_pt = 347.12354
    elif width == "AAAI":
        width_pt = 505.89
    elif width == "CVPR":
        width_pt = 496.85625 
    elif width == "CVPRSSingleCol":
        width_pt = 237.13594
    elif width == "AAAISingleCol":
        width_pt = 239.39438
    elif width == "NEURIPS":
        width_pt = 397.48499
    elif width == "ICCV":
        width_pt = 496.85625
    elif width == "ICCVSingleCol":
        width_pt = 237.13594
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    ratio

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def get_dataset_for_config(config):
    from src.classifier.inpainter import get_inpainter
    from improved_diffusion.image_datasets import ImageDataset
    all_files = pd.read_csv(config.data_dir)
    all_files = list(all_files["rel_path"])
    dataset = ImageDataset(
        os.path.dirname(config.data_dir),
        config.data.image_size,
        all_files,
    )
    inpainter = get_inpainter(config)
    inpaint_idx = dataset.inpaint_image(inpainter, config.data.saf.target_path)
    return dataset, inpaint_idx


def get_queryimage_for_config(config):
    dataset, i = get_dataset_for_config(config)
    return dataset[i]

def prepare_tdash_dataset(inputs, img_nums, M):
    """inputs are the private images. Each private image has to go through M forward passes. The img_nums are copied accordingly 
    to make retracing possible
    """
    private_images_input = []
    private_image_num = []
    for img, img_num in zip(inputs, img_nums): 
        private_images_input.append(repeat(img, "c h w -> b c h w", b=M))
        private_image_num.extend([img_num,]*M)
    private_images_input = torch.cat(private_images_input)
    return private_images_input, private_image_num
