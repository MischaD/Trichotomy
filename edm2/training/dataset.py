# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming images and labels from datasets created with dataset_tool.py."""
import pandas as pd
import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import random
import edm2.dnnlib
import os
import torch
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


try:
    import pyspng
except ImportError:
    pyspng = None

class ImageDataset(Dataset):
    def __init__(self, file_list, basedir, imagesize):
        self.basedir = basedir
        self.file_list = file_list
        self.imagesize = imagesize

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.basedir, self.file_list[idx])
        image = Image.open(img_path).convert('RGB')  # Ensure 3 channels
        image = image.resize((self.imagesize, self.imagesize))
        image = np.array(image).transpose(2, 0, 1)  # Convert to channel-first format for PyTorch
        image = torch.FloatTensor(image) / 255.0  # Normalize to [0,1] range
        return image, idx, self.file_list[idx]



class LatentDataset(Dataset):
    """
    A dataset that uses a FileList.csv to load a bunch of tensors. 
    Does not care whether the tensors are images or videos. If they are videos, randomly returns one index.
    """
    def __init__(self, filelist_txt, basedir, cond_mode="cond", load_to_memory=False):
        self.basedir = basedir
        self.file_list = []
        self.label_list = []
        self.load_to_memory = load_to_memory
        self.loaded_tensors = None
        self.condmode = cond_mode
        self.num_classes = -1

        # Parse the file list
        with open(filelist_txt, "r") as fp:
            for line in fp:
                if line.strip():  # Avoid empty lines

                    parts = line.split()
                    image_path = parts[0]
                    label = parts[1:]
                    if self.num_classes == -1: 
                        self.num_classes = len(label)

                    self.file_list.append(image_path)
                    self.label_list.append("".join(label).index("1"))

        # Initialize memory storage if loading into memory
        if self.load_to_memory:
            self.loaded_tensors = [None] * len(self.file_list)

        if self.condmode == "pseudocond": 
            self.pseudo_label_list = self.load_pseudolabels(feature_extractor="swav")

    def __len__(self):
        return len(self.file_list)
    
    def get_label(self, idx):
        if self.condmode == "uncond": 
            label = torch.zeros(self.num_classes) 
        elif self.condmode == "cond": 
            label = torch.zeros(self.num_classes) 
            label[self.label_list[idx]] = 1
        elif self.condmode == "pseudocond": 
            label = self.pseudo_label_list[idx] / torch.norm(self.pseudo_label_list[idx]) # normalize according to b.4 in edm paper
        else: 
            raise ValueError("Unknown conditioning mode")
        return label

    def __getitem__(self, idx):
        # Check if tensors are loaded to memory
        if self.load_to_memory:
            # If not already loaded, load it into memory
            if self.loaded_tensors[idx] is None:
                self.loaded_tensors[idx] = torch.load(os.path.join(self.basedir, self.file_list[idx]  + ".pt"))
            tensor = self.loaded_tensors[idx]
        else:
            # Load tensor directly from disk
            tensor = torch.load(os.path.join(self.basedir, self.file_list[idx] + ".pt"))

        label = self.get_label(idx)
        return tensor, idx, self.file_list[idx], label

    def load_pseudolabels(self, feature_extractor): 
        from beyondfid.data import hash_dataset_path
        feature_basedir = os.path.join(self.basedir, "FEATURES", feature_extractor)

        hash_filename = os.path.basename(hash_dataset_path(self.basedir, self.file_list, descriptor=feature_extractor)) + ".pt"
        if not os.path.exists(os.path.join(feature_basedir, hash_filename)):
            tmp_filelist = pd.DataFrame({"FileName":self.file_list, "Split":"TRAIN"})
            tmp_filelist.to_csv(".tmp_filelist.csv")
            print(f"hash data of feature extractor not found. Should be in: {os.path.join(feature_basedir, hash_filename)}")
            print(f"To fix we saved a tempory filelist for you as .tmp_filelist.csv. run beyondfid with the following command (you can ignore any error messages):")
            print(f"beyondfid .tmp_filelist.csv .tmp_filelist.csv  '' --output_path {os.path.dirname(feature_basedir)} --metrics 'fid' --feature_extractors {feature_extractor} --results_filename dummy.json --config-update=basedir=TODO/PATH/TO/BASEDIR/OF/DATASET")
            # saves features to dirname            
            exit(1)
            
        pseudo_label_list = torch.load(os.path.join(feature_basedir, hash_filename))
        return pseudo_label_list

class FeatureDataset(torch.utils.data.Dataset): 
    def __init__(self, path, data_base_path, pseudo_cond=False) -> None:
        self.csv_path = path
        self.paths = pd.read_csv(self.csv_path)["FileName"]
        self.data_base_path = data_base_path 
        self.pseudo_cond = pseudo_cond

        if self.data_base_path[-3:] != ".pt": 
            self.path_is_dir = True 
        else: 
            self.path_is_dir = False 
            print("Preloading dataset")
            self._preload_data = torch.load(self.data_base_path)
            assert len(self._preload_data) == len(self.paths), "Length of .pt feature file != the length of the .pt latent file."


        self.n_cond_features = None
        if self.pseudo_cond: 
            self._feature_data = torch.load(path[:-4] + ".pt")
            self.label_dim = self._feature_data.size()[-1]
        else: 
            feat = torch.load(path[:-4] + ".pt")
            self.label_dim = feat.size()[-1]
        self.num_channels = 4 

    def __getitem__(self, idx): 
        path = self.paths[idx]
        if self.path_is_dir: 
            data = torch.load(os.path.join(self.data_base_path, path) + ".pt")
        else: 
            # precomputed in large tensor
            data = self._preload_data[idx]

        if data.ndim == 4: 
            frame = random.randint(0, data.size()[0]-1)
            data = data[frame] # randomly train on one frame 

        if self.pseudo_cond: 
            feature = self._feature_data[idx]# if self.pseudo_cond else None
        else: 
            # uncondtional 
            feature = torch.zeros((self.label_dim,))

        return data, feature # (c x h x w), (l)

    def __len__(self): 
        return len(self.paths)

#----------------------------------------------------------------------------
# Abstract base class for datasets.

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        use_labels  = True,     # Enable conditioning labels? False = label dimension is zero.
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        cache       = False,    # Cache images in CPU memory?
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = dict() # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self._raw_shape[1:]
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self): # [CHW]
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = anything goes.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in supported_ext)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        ext = self._file_ext(fname)
        with self._open_file(fname) as f:
            if ext == '.npy':
                image = np.load(f)
                image = image.reshape(-1, *image.shape[-2:])
            elif ext == '.png' and pyspng is not None:
                image = pyspng.load(f.read())
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
            else:
                image = np.array(PIL.Image.open(f))
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------
