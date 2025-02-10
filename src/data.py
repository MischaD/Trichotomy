import os
import pandas as pd
from torch.utils.data import DataLoader
from utils import hash_dataset_path
import torch
import numpy as np
from src.latent import compute_latent_representation, get_latent_model
import os
import numpy as np
import torch 
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class ImageDatasetReal(Dataset):
    def __init__(self, root_dir, real_files, transform=None):
        """
        Args:
            root_dir (str): Root directory containing the image folders.
            transform (callable, optional): Transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = real_files

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_info = self.image_list[idx]

        full_path = os.path.join(self.root_dir, image_info['full_path'])
        image = Image.open(full_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {
            'image': image,
            'is_real': True,
            'model_name': image_info['model_name'],
            'class_name': image_info['class_name'],
            'full_path': full_path,
            'real_image_name': image_info['full_path']
        }


class SnthImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Root directory containing the image folders.
            transform (callable, optional): Transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = []
        self._load_images()

    def _load_images(self):
        # Traverse the directory and collect image information
        for model_name in os.listdir(self.root_dir):
            model_path = os.path.join(self.root_dir, model_name)
            if os.path.isdir(model_path):
                for class_name in os.listdir(model_path):
                    class_path = os.path.join(model_path, class_name)
                    if os.path.isdir(class_path):
                        images_path = os.path.join(class_path, 'images')
                        if os.path.isdir(images_path):
                            for image_name in os.listdir(images_path):
                                if image_name.endswith('.png'):
                                    # Extract real image name
                                    real_image_name = '_'.join(image_name.split('_')[:2]) + '.png'
                                    self.image_list.append({
                                        'model_name': model_name,
                                        'class_name': class_name,
                                        'real_image_name': real_image_name,
                                        'full_path': os.path.join(images_path, image_name)
                                    })

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_info = self.image_list[idx]
        image = Image.open(image_info['full_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {
            'image': image,
            'is_real': False,
            'model_name': image_info['model_name'],
            'class_name': image_info['class_name'].replace("_", " "), # No_finding --> No Finding 
            'full_path': image_info['full_path'],
            'real_image_name': image_info['real_image_name']
        }


# deprecation warning
#class LatentDatasetDeprecated(Dataset):
#    def __init__(self, file_list, basedir):
#        self.basedir = basedir
#        self.file_list = file_list
#
#    def __len__(self):
#        return len(self.file_list)
#
#    def __getitem__(self, idx):
#        img_path = os.path.join(self.basedir, self.file_list[idx]) + ".pt"
#        image = torch.load(img_path)
#        return image, idx, self.file_list[idx]
#
#
#class ImageDataset(Dataset):
#    def __init__(self, file_list, basedir, imagesize):
#        self.basedir = basedir
#        self.file_list = file_list
#        self.imagesize = imagesize
#
#    def __len__(self):
#        return len(self.file_list)
#
#    def __getitem__(self, idx):
#        img_path = os.path.join(self.basedir, self.file_list[idx])
#        image = Image.open(img_path).convert('RGB')  # Ensure 3 channels
#        image = image.resize((self.imagesize, self.imagesize))
#        image = np.array(image).transpose(2, 0, 1)  # Convert to channel-first format for PyTorch
#        image = torch.FloatTensor(image) / 255.0  # Normalize to [0,1] range
#        return image, idx, self.file_list[idx]


def get_data_from_txt(config):
    print(f"Reading File list from {config.filelist}")
    image_paths = []
    fileDescriptor = open(config.filelist, "r")
    line = True
    while line:
        line = fileDescriptor.readline()
        if line:
            lineItems = line.split()
            image_paths.append(lineItems[0])
    return image_paths, None


def get_data(config):
    data_csv = pd.read_csv(config.filelist)
    file_list = list(data_csv["FileName"])
    if hasattr(config, "debug") and config.debug: 
        file_list = file_list[:1000]
    output_filename = hash_dataset_path(os.path.dirname(config.filelist), "".join(file_list))
    return file_list, os.path.basename(output_filename)


def get_dataloader(file_list, config):
    basedir = config.basedir
    dataset = LatentDataset(file_list, basedir)  # Replaced VideoDataset with ImageDataset
    dataloader = DataLoader(dataset, batch_size=config.batch_size, num_workers=4, prefetch_factor=1)
    return dataloader


def get_distributed_image_dataloader(file_list, rank, world_size, config, base_name=None):
    if base_name is None: 
        base_name = os.path.dirname(config.filelist) if config.filelist.endswith(".csv") else config.filelist

    dataset = ImageDataset(file_list, base_name, imagesize=config.compute_latent.input_size)  # Replaced VideoDataset with ImageDataset
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=config.compute_latent.batch_size, sampler=sampler, num_workers=4, prefetch_factor=1)
    return dataloader


def get_data_from_folder(path):
    ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.pt')  # Adjusted to image extensions only
    # Walk through the directory
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(ALLOWED_EXTENSIONS):
                # Get the relative file path
                relative_path = os.path.relpath(os.path.join(root, file), path)
                file_list.append(relative_path)
    
    #output_filename = hash_dataset_path(path, img_list=file_list)
    return file_list#, os.path.basename(output_filename)
