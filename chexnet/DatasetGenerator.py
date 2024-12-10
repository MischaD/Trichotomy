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


class DatasetGenerator (Dataset):
    def __init__ (self, pathImageDirectory, pathDatasetFile, transform):
    
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform
    
        fileDescriptor = open(pathDatasetFile, "r")
        
        line = True
        
        while line:
                
            line = fileDescriptor.readline()
            
            if line:
          
                lineItems = line.split()
                
                imagePath = os.path.join(pathImageDirectory, lineItems[0])
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)   
            
        fileDescriptor.close()
    
    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel= torch.FloatTensor(self.listImageLabels[index])
        
        if self.transform != None: imageData = self.transform(imageData)
        
        return imageData, imageLabel
        
    def __len__(self):
        return len(self.listImagePaths)
    