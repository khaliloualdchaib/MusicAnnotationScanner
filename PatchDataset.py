import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import os

class PatchDataset(Dataset):
    def __init__(self, csv_file, transform=None) -> None:
        super().__init__()
        self.transform = transform
        self.df = pd.read_csv(csv_file)
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, patchID):
        filtered_df = self.df[(self.df["PatchID"] == patchID)]
        if filtered_df.empty:
            return "EMPTY"
        path = "Patches/" + filtered_df["OriginalImageID"].astype(str).iloc[0] + "/" + str(patchID) + ".png"
        img = io.imread(path)
        img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        return img, filtered_df["Annotation"].iloc[0]
    def ShowImage(self, Tensor_Image):
        # Convert the tensor to a numpy array
        image_array = Tensor_Image.numpy()
        plt.imshow(image_array)
        plt.show()
