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
        for j in tqdm(self.json["Training"], desc="Calculating mean and std of Training set"):
            img_path = path + j + ".png"
            img = io.imread(img_path)
            img = img[:, :, :3]
            tensor_img = transforms.ToTensor()(img)
            #print(tensor_img.shape)
            #print(torch.mean(tensor_img, dim=(1, 2)).tolist())
            mean_images.append(torch.mean(tensor_img, dim=(1, 2)).tolist())

        # Convert the list to a NumPy array
        mean_images_array = np.array(mean_images)

        # Calculate the mean and standard deviation
        mean = np.mean(mean_images_array, axis=0).tolist()
        std = np.std(mean_images_array, axis=0).tolist()
        mean.append(0)
        std.append(1)
        print("Mean (R, G, B):", mean)
        print("Standard Deviation (R, G, B):", std)

        """
        Mean (R, G, B): [0.8358980083480869, 0.7945440991078531, 0.6545946596475883]
        Standard Deviation (R, G, B): [0.07306775000426498, 0.06927377203989789, 0.09855596346528686]
        """
      
        for i in self.json:
            path = os.getcwd() + "\\" + i + "\\"
            for j in tqdm(self.json[i], desc="Normalizing " + str(i) + " set"):
                img_path = path + j + ".png"
                img = io.imread(img_path)
                #img = img[:, :, :3]
                tensor_img = transforms.ToTensor()(img)
                #mean=[0.485, 0.456, 0.406,0]
                #std=[0.229, 0.224, 0.225,1]
                normalized_tensor_img = transforms.Normalize(mean=mean, std=std)(tensor_img)
                pt_path = path + j + '.pt'
                torch.save(normalized_tensor_img, pt_path)
