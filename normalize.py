import json
import numpy as np
import torch
import os
from skimage import io

class Normalize:

    def __init__(self,DataSplitJSON) -> None:
         with open(DataSplitJSON, "r") as f:
            self.json = json.load(f)

    def normalize_images(self):
        for i in self.json:
            path = os.getcwd() + "\\" + i + "\\"
            for j in self.json[i]:
                img_path = path + j + ".png"
                img = io.imread(img_path)
                arr = np.array(img)
                """
                Linear normalization
                http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
                """
                # Do not touch the alpha channel
                for i in range(3):
                    minval = arr[...,i].min()
                    maxval = arr[...,i].max()
                    if minval != maxval:
                        arr[...,i] -= minval
                        arr[...,i] = arr[...,i] * (255.0/(maxval-minval))
                        
                arr.transpose(2,0,1)
                arr = torch.from_numpy(arr)
                pt_path = path + j + '.pt'
                torch.save(arr, pt_path)