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
    
    def CreatePatchesImage(self, image, index):
        json_file_name = "new.json"
        with open(json_file_name, "r") as f:
            json_file = json.load(f)

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

                annotation_label = None
                annotations_image = json_file["images"][index]["annotations"]
                if(len(annotations_image) is 0):
                    annotation_label = False
                else:
                    for annotation in annotations_image:
                        segmentations = annotation["segmentation"]
                        for i in range(0,len(segmentations)+1,2):
                            segmentation = [segmentation[i],segmentation[i+1]]
                            if self.is_segmentation_in_patch(segmentation,crop_left,crop_top,crop_right,crop_bottom):
                                annotation_label


                img = image[crop_left:crop_right, crop_top:crop_bottom]
                plt.imshow(img)
                filename = "patches/patch" + str(counter) + ".png"

                print(filename)
                print(annotation_label)
                
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
        for i in range(len(self.data)):
            self.CreatePatchesImage(self.data[i], i)

    def is_segmentation_in_patch(self,segmentation, patch_left, patch_top, patch_right, patch_bottom):
        for point in segmentation:
            if patch_left <= point[0] <= patch_right and patch_top <= point[1] <= patch_bottom:
                return True
        return False