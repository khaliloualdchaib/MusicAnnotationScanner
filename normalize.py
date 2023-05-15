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
        """
        Calculates mean and std of the training images and then normalizes all images in training,validation,testing based on the mean and std
        """
        path = os.getcwd() + "\\Training\\"
        mean_images = []
        std_images = []
        for j in tqdm(self.json["Training"], desc="Calculating mean and std of Training set"):
            img_path = path + j + ".png"
            img = io.imread(img_path)
            img = img[:, :, :3]
            mean_images.append([np.mean(img[:, :, 0]),np.mean(img[:, :, 1]),np.mean(img[:, :, 2])])
            #std_images.append([np.std(img[:, :, 0]),np.std(img[:, :, 1]),np.std(img[:, :, 2])])

        mean_images = np.array(mean_images)#, np.array(std_images)
        mean, std = np.mean(mean_images, axis=0).tolist(), np.std(mean_images, axis=0).tolist()
        print("Mean (R, G, B):", mean)
        print("Standard Deviation (R, G, B):", std)


        
        #print(mean)
        #print(std)
        
        """
        for i in self.json:
            path = os.getcwd() + "\\" + i + "\\"
            for j in tqdm(self.json[i], desc="Normalizing " + str(i) + " set"):
                img_path = path + j + ".png"
                img = io.imread(img_path)
                img = img[:, :, :3]
                tensor_img = transforms.ToTensor()(img)
                mean=[0.485, 0.456, 0.406]
                std=[0.229, 0.224, 0.225]
                normalized_tensor_img = transforms.Normalize(mean=mean, std=std)(tensor_img)
                pt_path = path + j + '.pt'
                torch.save(normalized_tensor_img, pt_path)
        """