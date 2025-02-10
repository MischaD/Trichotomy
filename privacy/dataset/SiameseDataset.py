from torch.utils import data
import pandas as pd
import numpy as np
import random
import os
from PIL import Image



class SiameseDataset(data.Dataset):
    def __init__(self, phase='training', n_channels=3, transform=None,
                 image_path='./', save_path=None, test_file="./image_pairs/TEST_pairs_all.txt"):

        self.phase = phase

        if self.phase == 'training':
            # In this way, images from one patient only appear in one subset.
            self.image_pairs = np.loadtxt('./image_pairs/TRAIN_pairs_all.txt', dtype=str)
        elif self.phase == 'validation':
            self.image_pairs = np.loadtxt('./image_pairs/VAL_pairs_all.txt', dtype=str)
        elif self.phase == 'testing':
            self.image_pairs = np.loadtxt(test_file, dtype=str)
        else:
            raise Exception('Invalid argument for parameter phase!')
        self.n_samples = len(self.image_pairs)

        self.n_channels = n_channels
        self.transform = transform

        # deprecated
        self.PATH = image_path 

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):

        x1 = pil_loader(self.image_pairs[index][0], self.n_channels)
        x2 = pil_loader(self.image_pairs[index][1], self.n_channels)

        if self.transform is not None:
            x1 = self.transform(x1)
            x2 = self.transform(x2)

        y1 = float(self.image_pairs[index][2])

        return x1, x2, y1


def pil_loader(path, n_channels):
    # TODO make this better.  
    if path[0] == "C": 
        base_dir="/vol/ideadata/ed52egek/data/chexpert/chexpertchestxrays-u20210408/"
    elif path[0] == "f": 
        base_dir="/vol/ideadata/ed52egek/data/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/"
    elif path[0] == "i": 
        base_dir="/vol/ideadata/ed52egek/data/chestxray14/"
    else: 
        ValueError("Unknown path")

    img = Image.open(os.path.join(base_dir, path)).convert('RGB')
    return img
        