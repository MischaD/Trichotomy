import sys
import os
import torch 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
from einops import repeat
from tqdm import tqdm
import pickle
import seaborn as sns
from PIL import Image
from torchvision import transforms


def load_latents(basepath, model_name, n_splits=1):
    """load .pt latent for <model_name>. Assumes only n_splits .pt for each model"""
    path = os.path.join(basepath, model_name)
    files = os.listdir(path)
    pts = [x for x in files if x.endswith(".pt") ]
    csv = [x for x in files if x.endswith(".csv")]
    assert len(pts) == n_splits and len(csv) == n_splits, f"Unexpected number of csv/pts found in {path}"

    feature_path = os.path.join(path, pts[0])
    print(f"Loading features in {feature_path}")
    features = torch.load(feature_path)
    paths = pd.read_csv(os.path.join(path, csv[0]))
    return features, paths


def update_matplotlib_font(fontsize=11, fontsize_ticks=8, tex=True, scale=1):
    import matplotlib.pyplot as plt
    fontsize = scale * fontsize
    fontsize_ticks = scale * fontsize_ticks
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


def export_image(img, image_pth): 
    # 0-1 img tensor c h w
    image = (img * 255).to(torch.uint8)
    image = image.permute(1, 2, 0).cpu().numpy()
    os.makedirs(os.path.dirname(image_pth), exist_ok=True)
    Image.fromarray(image, 'RGB').save(image_pth)


class_labels = ["No Finding", "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion", "Pneumonia", "Pneumothorax"] 


ds_to_viz = {
    "mimic": "MIMIC-CXR",
    "chexpert": "CheXpert",
    "cxr8": "ChestX-ray8"
}
