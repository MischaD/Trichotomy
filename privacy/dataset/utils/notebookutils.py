import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import os
from torch.utils import data
import torch
import pandas as pd
import numpy as np
import random
from PIL import Image
from einops import rearrange
import cv2
from utils import Utils
from tqdm import tqdm
from functools import partial
import shutil


def first_frame(vid): 
    return vid[0:1]

def subsample(vid, every_nth_frame): 
    frames = np.arange(0, len(vid), step=every_nth_frame)
    return vid[frames]


PHASE_TO_SPLIT = {"training": "TRAIN", "validation": "VAL", "testing": "TEST"}

def load_avi(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

class SimaseUSLatentDataset(data.Dataset):
    def __init__(self, 
                 phase='training', 
                 transform=None,
                 latents_csv='./', 
                 training_latents_base_path="./", 
                 in_memory=True, 
                 generator_seed=None):
        self.phase = phase
        self.training_latents_base_path = training_latents_base_path

        self.in_memory = in_memory
        self.videos = []

        self.df = pd.read_csv(latents_csv)
        self.df = self.df[self.df["Split"] == PHASE_TO_SPLIT[self.phase]].reset_index(drop=True)

        self.transform = transform

        if generator_seed is None: 
            self.generator = np.random.default_rng() 
            #unseeded
        else:             
            self.generator_seed = generator_seed
            print(f"Set {self.phase} dataset seed to {self.generator_seed}")

        if self.in_memory: 
            self.load_videos()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        vid_a = self.get_vid(index)
        if self.transform is not None:
            vid_a = self.transform(vid_a)
        return vid_a

    def reset_generator(self): 
        self.generator = np.random.default_rng(self.generator_seed) 

    def get_vid(self, index, from_disk=False): 
        if self.in_memory and not from_disk: 
            return self.videos[index]
        else: 
            return torch.load(os.path.join(self.training_latents_base_path, self.df.iloc[index]["FileName"] + ".pt"))


class SimaseUSVideoDataset(data.Dataset):
    def __init__(self, 
                 phase='training', 
                 transform=None,
                 latents_csv='./', 
                 training_latents_base_path="./", 
                 in_memory=True, 
                 generator_seed=None):
        self.phase = phase
        self.training_latents_base_path = training_latents_base_path

        self.in_memory = in_memory
        self.videos = []

        self.df = pd.read_csv(latents_csv)
        self.df = self.df[self.df["Split"] == PHASE_TO_SPLIT[self.phase]].reset_index(drop=True)

        self.transform = transform

        if generator_seed is None: 
            self.generator = np.random.default_rng() 
            #unseeded
        else:             
            self.generator_seed = generator_seed
            print(f"Set {self.phase} dataset seed to {self.generator_seed}")

        if self.in_memory: 
            self.load_videos()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        vid_a = self.get_vid(index)
        if self.transform is not None:
            vid_a = self.transform(vid_a)
        return vid_a

    def reset_generator(self): 
        self.generator = np.random.default_rng(self.generator_seed) 

    def load_videos(self): 
        self.videos = []
        print("Preloading videos")
        for i in range(len(self)):
            self.videos.append(self.get_vid(i, from_disk=True))

    def get_vid(self, index, from_disk=False): 
        if self.in_memory and not from_disk: 
            return self.videos[index]
        else: 
            file_name = self.df.iloc[index]["FileName"] + ".avi" if not self.df.iloc[index]["FileName"].endswith(".avi") else self.df.iloc[index]["FileName"]
            video_file_path = os.path.join(self.training_latents_base_path, file_name)
            video_frames = load_avi(video_file_path)
            video_frames = torch.tensor(np.stack(video_frames))
            video_frames = video_frames / 127.5 - 1 
            return rearrange(video_frames, "b h w c -> b c h w")


class SiameseNetwork(nn.Module):
    def __init__(self, network='ResNet-50', in_channels=3, n_features=128):
        super(SiameseNetwork, self).__init__()
        self.network = network
        self.in_channels = in_channels
        self.n_features = n_features

        if self.network == 'ResNet-50':
            # Model: Use ResNet-50 architecture
            self.model = models.resnet50(pretrained=True)
            # Adjust the input layer: either 1 or 3 input channels
            if self.in_channels == 1:
                self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            if self.in_channels == 4: 
                self.model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            elif self.in_channels == 3:
                pass
            else:
                raise Exception(
                    'Invalid argument: ' + self.in_channels + '\nChoose either in_channels=1 or in_channels=3')
            # Adjust the ResNet classification layer to produce feature vectors of a specific size
            self.model.fc = nn.Linear(in_features=2048, out_features=self.n_features, bias=True)

        else:
            raise Exception('Invalid argument: ' + self.network +
                            '\nChoose ResNet-50! Other architectures are not yet implemented in this framework.')

        self.fc_end = nn.Linear(self.n_features, 1)

    def forward_once(self, x):

        # Forward function for one branch to get the n_features-dim feature vector before merging
        output = self.model(x)
        output = torch.sigmoid(output)
        return output

    def prediction_head_forward(self, output1, output2):
        difference = torch.abs(output1 - output2)
        output = self.fc_end(difference)
        return output

    def forward(self, input1, input2):

        # Forward
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # prediction head
        output = self.prediction_head_forward(output1, output2)
        return output


def model_forward_to_corrcoeff(model, video, bs=256):
    latents_train = []
    with torch.no_grad():
        for i in np.arange(0, len(video), bs):
            batch = video[i:i+bs].cuda()
            latents_train.append(model.forward_once(batch))

    latents_train = torch.cat(latents_train)
    train_val_corr_orig = torch.corrcoef(torch.cat([latents_train])).cpu()
    return train_val_corr_orig

def model_forward_to_pred(model, input_a, input_b, bs=256):
    assert len(input_a) == len(input_b)
    predictions = []
    with torch.no_grad():
        for i in np.arange(0, len(input_a), bs):
            batch_a = input_a[i:i+bs].cuda()
            batch_b = input_b[i:i+bs].cuda()
            pred = model.forward(batch_a, batch_b)[:, 0]
            pred = torch.sigmoid(pred)
            predictions.append(pred.cpu())
    predictions = torch.cat(predictions)
    return predictions 

def model_forward_to_bin_pred(model, input_a, input_b, bs=256):
    pred = model_forward_to_pred (model, input_a, input_b, bs)
    pred = Utils.apply_threshold(pred.numpy(), 0.5)
    return pred 


def model_forward_to_corr_coeff(model, input_a, input_b, bs=256): 
    coeffs = []
    with torch.no_grad():
        for i in np.arange(0, len(input_a), bs):
            batch_a = input_a[i:i+bs].cuda()
            batch_b = input_b[i:i+bs].cuda()
            feature_a = model.forward_once(batch_a)
            feature_b = model.forward_once(batch_b)
        coeffs.append(torch.corrcoef(torch.cat([feature_a, feature_b])).cpu())
    coeffs = torch.cat(coeffs)
    return coeffs


def model_forward_to_corr_coeff(model, input_a, input_b, bs=256): 
    # two single frame videos --> corr coeff according to model 
    coeffs = []
    with torch.no_grad():
        for i in np.arange(0, len(input_a), bs):
            batch_a = input_a[i:i+bs].cuda()
            batch_b = input_b[i:i+bs].cuda()
            feature_a = model.forward_once(batch_a)
            feature_b = model.forward_once(batch_b)
        coeffs.append(torch.corrcoef(torch.cat([feature_a, feature_b])).cpu()[0, 1])
    coeffs = torch.stack(coeffs)
    return coeffs


def create_new_dataset(new_full_ds_syn, out_dir,real_data_base_path="", syn_data_base_path="", real_splits=["TEST"], allow_dropping=False, video_subdir="avi"):
    # copies all videos from new_full_ds_syn to out_dir/Videos and creates videos
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir + "/Videos", exist_ok=True)
    print(f"Saving new dataset to {out_dir}")

    print(f"{new_full_ds_syn.Split.value_counts()}")

    files_dropped = 0
    for filename, split in tqdm(zip(new_full_ds_syn["FileName"], new_full_ds_syn["Split"]), "Copying files to new training dir", total=len(new_full_ds_syn)): 
        try: 
            if split in real_splits: 
                # base dir is real videos
                shutil.copy(os.path.join(real_data_base_path, f"{filename}.avi"), os.path.join(out_dir, f"Videos/{filename}.avi"))
            else: 
                shutil.copy(os.path.join(syn_data_base_path, f"{video_subdir}/{filename}"), os.path.join(out_dir, f"Videos/{filename}"))

        except FileNotFoundError as e: 
            if allow_dropping: 
                files_dropped += 1
                new_full_ds_syn = new_full_ds_syn[new_full_ds_syn['FileName'] != filename]
            else: 
                raise FileNotFoundError(e)

    if files_dropped > 0: 
        print(f"Warning: {files_dropped} files dropped")
    print(f"{new_full_ds_syn.Split.value_counts()}")
    new_full_ds_syn.to_csv(os.path.join(out_dir, "FileList.csv"))