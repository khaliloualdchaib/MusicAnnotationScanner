from Dataloader import *
from Rescale import *
import matplotlib.pyplot as plt
import pandas as pd
import json


class Patch:
    def __init__(self, data, height, width) -> None:
        self.data = data
        self.imageheight, self.imagewidth = self.data[0].shape[1], self.data[0].shape[0]
        self.height = height
        self.width = width
        self.csv = {
            'Name':[], 
            'PatchURL': [], 
            'TopleftX': [], 
            'TopleftY': [], 
            'BottomrightX': [],
            'BottomrightY': []
        }
    
    def CreatePatchesImage(self, image):
        counter = 0
        for w in range(0, self.imagewidth, self.width):
            for h in range(0, self.imageheight, self.height):
                if w + self.width >= self.imagewidth and h + self.height >= self.imageheight:
                    overlapheight = (h+self.height) - self.imageheight
                    overlapwidth = (w+self.width) - self.imagewidth
                    crop_right = self.imagewidth
                    crop_bottom = self.imageheight
                    crop_left = w - overlapwidth
                    crop_top = h - overlapheight
                elif w + self.width >= self.imagewidth:
                    overlap = (w+self.width) - self.imagewidth
                    crop_right = self.imagewidth
                    crop_bottom = h + self.height
                    crop_left = w - overlap
                    crop_top = h
                elif h + self.height >= self.imageheight:
                    overlap = (h+self.height) - self.imageheight
                    crop_right = w + self.width
                    crop_bottom = self.imageheight
                    crop_left = w
                    crop_top = h - overlap
                else:
                    crop_right = w + self.width
                    crop_bottom = h + self.height
                    crop_left = w
                    crop_top = h
                img = image[crop_left:crop_right, crop_top:crop_bottom]
                plt.imshow(img)
                filename = "patches/patch" + str(counter) + ".png"
                counter += 1
                plt.savefig(filename)
                self.csv["Name"].append("patch" + str(counter))
                self.csv["PatchURL"].append("patches/patch" + str(counter) + ".png")
                self.csv["TopleftX"].append(crop_left)
                self.csv["TopleftY"].append(crop_top)
                self.csv["BottomrightX"].append(crop_right)
                self.csv["BottomrightY"].append(crop_bottom)
        df = pd.DataFrame(self.csv)
        df.to_csv('patches.csv')

    def CreatePatches(self):
        for i  in range(len(self.data)):
            self.cropcoordinates(self.data[i], i)