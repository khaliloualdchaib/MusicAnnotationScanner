from FAAMDataset import *
from toTesnor import ToTensor
from patches import *
from PatchDataset import *

import csv # for def Datasplitting
import math # for def Datasplitting
import random # for def Datasplitting

#Put here the path to root folder
root = os.getcwd()
dataset = FAAMDataset("new.json")
#dataset.transform = transform=transforms.Compose([Rescale(dataset.getAverageSize())])
patch_width = 500
patch_height = 500
#p = Patch(dataset, patch_height, patch_width, root)
#p.CreatePatches()
#p.CreatePatchesImage(dataset[3],3)
#print(p)
#p.ShowImage(patches["image"], True, "ShowImageTest.png")

data = PatchDataset("Patches.csv")

#data.ShowImage(data[0][0])
"""
clean = 0
not_clean = 0

for i in range(len(data)):
    print(data[i][1])
    if data[i][1] == 0:
        clean +=1
        print("Clean")
    else:
        not_clean+=1
        print("Annotated")



print(clean)
print(not_clean)
"""


def Datasplitting(trainingpercentage, testingpercentage, validationpercentage,patchdataset, dataset, cleansize = 8504, annotatedsize = 2962):
    totalsize = cleansize + annotatedsize # totalsize of patches
    validationsize = floor(totalsize * validationpercentage)
    testingsize = floor(totalsize * testingpercentage)
    trainingsize = floor(totalsize * trainingpercentage)
    images = len(dataset)
    print("totalsize", totalsize)
    print("trainingsize",trainingsize)
    print("testingsize",testingsize)
    print("validationsize",validationsize)
    print("images",images)

    # Create lists this should be made in patches.py for less computation

    clean = {}
    annotated = {}

    with open('patches.csv', 'r') as file: # need to get as parameter
        reader = csv.DictReader(file)
        for row in reader:
            if row["Annotation"] == "True":
                if(row["OriginalImageID"] in annotated):
                    annotated[row["OriginalImageID"]].append(row["PatchID"])
                else:
                    annotated[row["OriginalImageID"]] = [row["PatchID"]]
            else:
                if(row["OriginalImageID"] in clean):
                    clean[row["OriginalImageID"]].append(row["PatchID"])
                else:
                    clean[row["OriginalImageID"]] = [row["PatchID"]]

    #print("annotated",annotated)


    

    Training_dataset = {}
    Testing_dataset = {}
    Validation_dataset = {}

    #### CREATE TRAINING DATASET
    for i in clean:    
        amount_patches_vertical = dataset[int(i)].shape[1] / patch_height
        amount_patches_horizontal = dataset[int(i)].shape[0] / patch_width
        patch_amount = math.ceil(amount_patches_vertical) * math.ceil(amount_patches_horizontal)
        amount_per_image = int(patch_amount * trainingpercentage)
    
        random.shuffle(clean[i])
        Training_dataset[i] = clean[i][:amount_per_image]
        clean[i] = clean[i][amount_per_image:]
        
    #### CREATE TESTING,VALIDATION DATASET


    for i in clean:
        amount_patches_vertical = dataset[int(i)].shape[1] / patch_height
        amount_patches_horizontal = dataset[int(i)].shape[0] / patch_width
        patch_amount = math.ceil(amount_patches_vertical) * math.ceil(amount_patches_horizontal)
        amount_per_image_testing = int(patch_amount * testingpercentage)
        amount_per_image_validation = int(patch_amount * validationpercentage)

        random.shuffle(annotated[i])
        Testing_dataset[i] = clean[i][:amount_per_image_testing]
        Testing_dataset[i].extend(annotated[i][:amount_per_image_testing])
        clean[i] = clean[i][amount_per_image_testing:]
        annotated[i] = clean[i][amount_per_image_testing:]

        Validation_dataset[i] = clean[i][:amount_per_image_validation]
        Validation_dataset[i].extend(annotated[i][:amount_per_image_validation])
        clean[i] = clean[i][amount_per_image_validation:]
        annotated[i] = clean[i][amount_per_image_validation:]


    training_path = os.getcwd() + "\\Training\\"
    testing_path = os.getcwd() + "\\Testing\\"
    validation_path = os.getcwd() + "\\Validation\\"
    Patches_path = os.getcwd() + "\\Patches\\"
    if not os.path.exists(training_path):
        os.makedirs(training_path)
    if not os.path.exists(testing_path):
        os.makedirs(testing_path)
    if not os.path.exists(validation_path):
        os.makedirs(validation_path)
    for i in Training_dataset:
        for j in Training_dataset[i]:
            source = Patches_path + i + "\\" + j + ".png"
            destination = training_path + j + ".png"
            os.rename(source,destination)
    for i in Testing_dataset:
        for j in Testing_dataset[i]:
            source = Patches_path + i + "\\" + j + ".png"
            destination = testing_path + j + ".png"
            os.rename(source,destination)
    for i in Validation_dataset:
        for j in Validation_dataset[i]:
            source = Patches_path + i + "\\" + j + ".png"
            destination = validation_path + j + ".png"
            os.rename(source,destination)







Datasplitting(0.40,0.40,0.20,data,dataset)




