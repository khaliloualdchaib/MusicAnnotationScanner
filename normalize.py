import json
import numpy as np
import torch
import os
from skimage import io
import torchvision.transforms as transforms
from tqdm import tqdm

class Normalize:
    def __init__(self,DataSplitJSON) -> None:
         with open(DataSplitJSON, "r") as f:
            self.json = json.load(f)
    def normalize_images(self):
        for i in self.json:
            path = os.getcwd() + "\\" + i + "\\"
            for j in tqdm(self.json[i], desc="Normalizing " + str(i) + " set"):
                img_path = path + j + ".png"
                img = io.imread(img_path)
                tensor_img = transforms.ToTensor()(img)
                mean=[0.485, 0.456, 0.406,0.0]
                std=[0.229, 0.224, 0.225,1.0]
                normalized_tensor_img = transforms.Normalize(mean=mean, std=std)(tensor_img)
                pt_path = path + j + '.pt'
                torch.save(normalized_tensor_img, pt_path)