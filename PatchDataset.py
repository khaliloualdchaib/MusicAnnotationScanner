import torch
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import json

class PatchDataset(Dataset):
    def __init__(self, csv_file, DataSplitJSON, type, transform=None) -> None:
        super().__init__()
        self.transform = transform
        self.df = pd.read_csv(csv_file)
        self.type = type
        with open(DataSplitJSON, "r") as f:
            self.json = json.load(f)
    def __len__(self):
        return len(self.json[self.type])
    
    def __getitem__(self, idx):
        patchID = int(self.json[self.type][idx])
        filtered_df = self.df[(self.df["PatchID"] == patchID)]
        if filtered_df.empty:
            return "EMPTY"    
        path = self.type + "/" + str(patchID) + ".png"
        img = io.imread(path)
        img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        return img, filtered_df["Annotation"].iloc[0]
def ShowImage(self, Tensor_Image):
        # Convert the tensor to a numpy array
        image_array = Tensor_Image.numpy()
        plt.imshow(image_array)
        plt.show()
