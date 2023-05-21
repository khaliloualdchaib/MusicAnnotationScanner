import json
import numpy as np
import torch
import os
from skimage import io
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image


class Normalize:
    def __init__(self,DataSplitJSON) -> None:
         with open(DataSplitJSON, "r") as f:
            self.json = json.load(f)

    def get_mean_std(self):
        path = os.getcwd() + "\\Training\\"
        mean = np.zeros(3)
        std = np.zeros(3)
        num_samples = len(self.json["Training"])
        for j in tqdm(self.json["Training"], desc="Calculating mean and std of Training set"):
            img_path = path + j + ".png"
            image = Image.open(img_path)
            rgb_image = image.convert("RGB")
            tensor_image = np.array(rgb_image) / 255.0
            mean += np.mean(tensor_image, axis=(0, 1))
            std += np.std(tensor_image, axis=(0, 1))

        
        mean /= num_samples
        std /= num_samples
        print("Mean (R, G, B):", mean)
        print("Standard Deviation (R, G, B):", std)
        return mean, std
        """
        channels_sum = torch.zeros(3)
        channels_squared_sum = torch.zeros(3)

        path = os.getcwd() + "\\Training\\"
        for j in tqdm(self.json["Training"], desc="Calculating mean and std of Training set"):
            img_path = path + j + ".png"
            img = io.imread(img_path)
            img = img[:, :, :3]
            image = transforms.ToTensor()(img)
            channels_sum += torch.mean(image, dim=[1, 2])
            channels_squared_sum += torch.mean(image ** 2, dim=[1, 2])
        
        num_images = len(self.json["Training"])
        mean = channels_sum / num_images
        std = (channels_squared_sum / num_images - mean ** 2) ** 0.5
        #print("Mean (R, G, B):", mean)
        #print("Standard Deviation (R, G, B):", std)
        print("Mean (R, G, B):", mean)
        print("Standard Deviation (R, G, B):", std)
        return mean, std
        """
    def normalize_images(self):
        values = self.get_mean_std()
        mean = values[0]
        std = values[1]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        for i in self.json:
            path = os.getcwd() + "\\" + i + "\\"
            for j in tqdm(self.json[i], desc="Normalizing " + str(i) + " set"):
                img_path = path + j + ".png"
                image = Image.open(img_path)
                rgb_image = image.convert("RGB")
                normalized_tensor_img = transform(rgb_image)
                pt_path = path + j + '.pt'
                torch.save(normalized_tensor_img, pt_path)



"""
old normalize function    
 def normalize_images(self):
        path = os.getcwd() + "\\Training\\"
        mean_images = []        
        for j in tqdm(self.json["Training"], desc="Calculating mean and std of Training set"):
            img_path = path + j + ".png"
            img = io.imread(img_path)
            img = img[:, :, :3]
            tensor_img = transforms.ToTensor()(img)
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
        Mean (R, G, B): [0.8358980083480869, 0.7945440991078531, 0.6545946596475883]
        Standard Deviation (R, G, B): [0.07306775000426498, 0.06927377203989789, 0.09855596346528686]
      
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
    """
