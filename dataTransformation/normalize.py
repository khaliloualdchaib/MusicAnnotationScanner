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
        self.transform = transforms.ToTensor()

    def get_mean_std(self):
        total_pixels = 500*500*len(self.json["Training"])
        pixels_in_image = 500*500
        red_mean = 0
        green_mean = 0
        blue_mean = 0
        red_var = 0
        green_var = 0
        blue_var = 0
        path = os.getcwd() + "\\Training\\"
        for j in tqdm(self.json["Training"], desc="Calculating mean of Training set"):
            img_path = path + j + ".png"
            image = Image.open(img_path)
            rgb_image = image.convert("RGB")
            tensor_image = self.transform(rgb_image)

            red_channel = tensor_image[0].flatten()
            green_channel = tensor_image[1].flatten()
            blue_channel = tensor_image[2].flatten()

            red_mean += torch.sum(red_channel)
            green_mean += torch.sum(green_channel)
            blue_mean += torch.sum(blue_channel)
            
        red_mean /= total_pixels
        green_mean /= total_pixels
        blue_mean /= total_pixels

        for j in tqdm(self.json["Training"], desc="Calculating std of Training set"):
            img_path = path + j + ".png"
            image = Image.open(img_path)
            rgb_image = image.convert("RGB")
            tensor_image = self.transform(rgb_image)

            red_channel = tensor_image[0].flatten()
            green_channel = tensor_image[1].flatten()
            blue_channel = tensor_image[2].flatten()

            red_var += torch.sum(torch.square(red_channel - red_mean))
            green_var += torch.sum(torch.square(green_channel - green_mean))
            blue_var += torch.sum(torch.square(blue_channel - blue_mean))            

        red_var /= total_pixels
        green_var /= total_pixels
        blue_var /= total_pixels

        red_std = np.sqrt(red_var)
        green_std = np.sqrt(green_var)
        blue_std = np.sqrt(blue_var)

        means = torch.tensor([red_mean, green_mean, blue_mean])
        stds = torch.tensor([red_std, green_std, blue_std])

        print("Mean (R, G, B):", means)
        print("Standard Deviation (R, G, B):", stds)
        return means, stds
    def normalize_images(self):
        #values = self.get_mean_std()
        #mean = values[0]
        #std = values[1]
        for i in self.json:
            path = os.getcwd() + "\\" + i + "\\"
            for j in tqdm(self.json[i], desc="Normalizing " + str(i) + " set"):
                img_path = path + j + ".png"
                image = Image.open(img_path)
                rgb_image = image.convert("RGB")
                tensor_image = self.transform(rgb_image)
                #print("Minimum value of tensor_image:", tensor_image.min())
                #print("Maximum value of tensor_image:", tensor_image.max())
                normalized_tensor_img = transforms.Normalize(mean=0, std=1)(tensor_image)
                #print("Minimum value of normalized_tensor_img:", normalized_tensor_img.min())
                #print("Maximum value of normalized_tensor_img:", normalized_tensor_img.max())
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
