import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


#class DatasetGenerator(Dataset):
#    def __init__(self, pathImageDirectory, pathDatasetFile, transform):
#        self.transform = transform
#        self.cachedData = []
#        
#        # Load paths and labels from the dataset file
#        with open(pathDatasetFile, "r") as fileDescriptor:
#            for line in fileDescriptor:
#                if line.strip():
#                    lineItems = line.split()
#                    imagePath = os.path.join(pathImageDirectory, lineItems[0])
#                    imageLabel = [int(i) for i in lineItems[1:]]
#                    self.cachedData.append((imagePath, imageLabel))
#
#        self.cachedData = self.cachedData
#        # Preload all images into memory
#        print("Preloading Dataset...")
#        self.cachedData = [
#            (self._load_image(imagePath), torch.FloatTensor(imageLabel))
#            for imagePath, imageLabel in tqdm(self.cachedData, desc="Loading Images")
#        ]
#        print("Preloading dataset finished.")
#    
#    def _load_image(self, imagePath):
#        imageData = Image.open(imagePath).convert('RGB')
#        return imageData
#
#    def __getitem__(self, index):
#        imageData = self.cachedData[index]
#        if self.transform is not None:
#            imageData = self.transform(imageData[0]), imageData[1]
#        return imageData
#    
#    def __len__(self):
#        return len(self.cachedData)


import os
from torch.utils.data import Dataset
from PIL import Image
import torch

class DatasetGenerator(Dataset):
    def __init__(self, pathImageDirectory, pathDatasetFile, transform, use_cache=False):
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform
        self.use_cache = use_cache
        self.image_cache = {}  # Cache to store loaded images
        self.replace_with_png = False

        with open(pathDatasetFile, "r") as fileDescriptor:
            for line in fileDescriptor:
                if line.strip():
                    lineItems = line.split()
                    imagePath = os.path.join(pathImageDirectory, lineItems[0])
                    imageLabel = list(map(int, lineItems[1:]))
                    self.listImagePaths.append(imagePath)
                    self.listImageLabels.append(imageLabel)

    def check_jpg_vs_png(self, image_path):
        if os.path.exists(image_path):
            return False
        if os.path.exists(image_path[:-4] + ".png"):
            return True
        raise FileNotFoundError(f"File not found (also not the PNG version): {image_path} or {image_path[:-4] + '.png'}")

    def load_image(self, image_path):
        if image_path not in self.image_cache:
            img = Image.open(image_path).convert('RGB').resize((512, 512))
            if self.use_cache == True: 
                self.image_cache[image_path] = img
            else: 
                return img
        return self.image_cache[image_path]

    def __getitem__(self, index):
        imagePath = self.listImagePaths[index]

        # Check file extension replacement only once
        if index == 0:
            self.replace_with_png = self.check_jpg_vs_png(imagePath)

        if self.replace_with_png:
            imagePath = imagePath[:-4] + ".png"

        imageData = self.load_image(imagePath)
        imageLabel = torch.FloatTensor(self.listImageLabels[index])

        if self.transform:
            imageData = self.transform(imageData)

        return imageData, imageLabel

    def __len__(self):
        return len(self.listImagePaths)
